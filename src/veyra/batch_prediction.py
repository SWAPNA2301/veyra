"""Bounded, in-memory CSV ingestion. Uploaded values never enter logs or caches."""

import csv
from email import policy
from email.parser import BytesHeaderParser
from io import StringIO
import math
import re

MAX_CSV_BYTES = 5_000_000
MAX_REQUEST_BYTES = MAX_CSV_BYTES + 16_384  # Bounded multipart framing allowance.
MAX_ROWS = 5000
NUMERIC = {"TransactionAmt", "dist1", "dist2"}
FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)


class InvalidBatch(ValueError):
    """Only application-authored, input-free error messages may be exposed."""

    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def csv_payload(body: bytes, content_type: str) -> bytes:
    """Accept raw text/csv or exactly one multipart file field, without spooling.

    Parse only bounded MIME headers, not an arbitrary MIME message tree. Multipart
    boundaries cannot occur in a valid part's content; extra parts are rejected.
    """
    if len(body) > MAX_REQUEST_BYTES:
        raise InvalidBatch("Upload exceeds the 5 MB CSV limit.", 413)
    media = content_type.split(";", 1)[0].strip().lower()
    if media == "text/csv":
        payload = body
    elif media == "multipart/form-data":
        try:
            if len(content_type) > 1024 or "\r" in content_type or "\n" in content_type:
                raise ValueError
            parser = BytesHeaderParser(policy=policy.default.clone(raise_on_defect=True))
            envelope = parser.parsebytes(b"Content-Type: " + content_type.encode("ascii") + b"\r\n\r\n")
            boundary = envelope.get_boundary()
            if not boundary or not 1 <= len(boundary) <= 70 or not re.fullmatch(r"[0-9A-Za-z'()+_,./:=? -]+", boundary) or boundary.endswith(" "):
                raise ValueError
            marker = b"--" + boundary.encode("ascii")
            if not body.startswith(marker + b"\r\n"):
                raise ValueError
            parts = body[len(marker) + 2:].split(b"\r\n" + marker)
            if len(parts) != 2 or parts[1] not in (b"--", b"--\r\n"):
                raise ValueError
            headers, separator, payload = parts[0].partition(b"\r\n\r\n")
            if not separator or len(headers) > 8192:
                raise ValueError
            part = parser.parsebytes(headers + b"\r\n\r\n")
            if (len(part.get_all("Content-Disposition", [])) != 1
                    or part.get_content_disposition() != "form-data"
                    or part.get_param("name", header="Content-Disposition") != "file"
                    or not (part.get_filename() or "").lower().endswith(".csv")
                    or part.get("Content-Transfer-Encoding") is not None):
                raise ValueError
        except Exception:
            raise InvalidBatch("Upload exactly one .csv file in multipart field 'file', with valid multipart framing.", 400) from None
    else:
        raise InvalidBatch("Use multipart/form-data with field 'file', or text/csv.", 415)
    if len(payload) > MAX_CSV_BYTES:
        raise InvalidBatch("CSV exceeds 5 MB (5,000,000 bytes).", 413)
    return payload


def validated_rows(payload: bytes, schema):
    """Strict UTF-8 CSV; data-row numbers are one-based, excluding the header."""
    if len(payload) > MAX_CSV_BYTES:
        raise InvalidBatch("CSV exceeds 5 MB (5,000,000 bytes).", 413)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeError:
        raise InvalidBatch("CSV must be UTF-8 encoded (an optional UTF-8 BOM is supported).") from None
    if not text.strip() or "\x00" in text:
        raise InvalidBatch("CSV is empty or contains invalid NUL characters.")
    reader = csv.reader(StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
        header = [name.strip() for name in header]
        if len(header) != len(set(header)):
            raise InvalidBatch("Duplicate column names are not allowed (including whitespace-normalized duplicates).")
        allowed = set(schema.model_fields)
        if not header or any(not name or name not in allowed for name in header):
            raise InvalidBatch("CSV contains empty or unsupported columns. Allowed columns: " + ", ".join(schema.model_fields) + ".")
        if "TransactionAmt" not in header:
            raise InvalidBatch("CSV requires a TransactionAmt column.")
        rows = []
        for index, cells in enumerate(reader, 1):
            if index > MAX_ROWS:
                raise InvalidBatch("CSV exceeds 5000 data rows.", 413)
            if len(cells) != len(header):
                raise InvalidBatch(f"Data row {index} must have exactly {len(header)} fields; blank rows are not allowed.")
            values = {}
            for name, raw in zip(header, cells):
                value = raw.strip()
                if not value:
                    if name == "TransactionAmt":
                        raise InvalidBatch(f"Data row {index}: TransactionAmt is required and cannot be blank.")
                    values[name] = None
                elif name in NUMERIC:
                    if not NUMBER.fullmatch(value):
                        raise InvalidBatch(f"Data row {index}: {name} must be a finite nonnegative number.")
                    number = float(value)
                    if not math.isfinite(number) or not 0 <= number <= FLOAT32_MAX:
                        raise InvalidBatch(f"Data row {index}: {name} must be nonnegative, finite and within float32 range.")
                    values[name] = number
                else:
                    if len(value) > 128:
                        raise InvalidBatch(f"Data row {index}: {name} must contain at most 128 characters.")
                    values[name] = value
            rows.append(schema.model_validate(values).model_dump())
    except InvalidBatch:
        raise
    except (csv.Error, StopIteration, ValueError):
        raise InvalidBatch("Malformed CSV or invalid field types. Check comma delimiters, quotes, and the required numeric amount.") from None
    if not rows:
        raise InvalidBatch("CSV must contain at least one data row after the header.")
    return rows


def score_rows(rows, spec):
    """One vectorized prediction using the unchanged fitted preprocessing."""
    import numpy as np
    import pandas as pd
    from .dashboard.predictor import load_fitted_pipeline

    pipeline = load_fitted_pipeline(spec)
    scores = np.asarray(pipeline.predict_proba(pd.DataFrame(rows, columns=spec.selected)))
    if scores.shape != (len(rows), 2) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid classifier output")
    results = [{"row_number": i, "score": float(score), "flagged": bool(score >= spec.threshold)}
               for i, score in enumerate(scores[:, 1], 1)]
    flagged = sum(row["flagged"] for row in results)
    return {
        "rows": results,
        "counts": {"total": len(rows), "flagged": flagged, "not_flagged": len(rows) - flagged},
        "threshold": spec.threshold, "model_run_id": spec.run_id,
        "score_type": "uncalibrated",
        "notice": "Scores are uncalibrated model outputs, not fraud probabilities. Flagged means at or above the frozen validation threshold, not confirmed fraud.",
    }
