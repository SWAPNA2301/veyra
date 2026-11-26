"""Reviewed aggregate-only corpus. No recursive discovery or link following."""

import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import re

CATALOG = Path(__file__).with_name("copilot_catalog.json")
MAX_SOURCE_BYTES = 200_000
SHAP_FIELDS = (
    "validation_only", "sample_rows", "seed", "sampling", "encoded_columns",
    "original_features", "boosting_iteration", "method", "units", "precision",
    "grouping", "beeswarm_color", "max_additivity_error", "preprocessor_unchanged",
    "model_file_unchanged", "retrained", "test_used", "labels_used", "interpretation",
)
SECRET = re.compile(r"-----BEGIN .*PRIVATE KEY|\b(?:sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b|(?:password|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"]?[^\s'\"]{8,}", re.I)


class CorpusUnavailable(ValueError):
    """Safe, non-sensitive failure; callers never return the underlying input."""


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def local_path(root, relative):
    """Reject links/junctions as well as lexical and resolved path traversal."""
    root = root.resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise CorpusUnavailable("Source is outside the approved local corpus.")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise CorpusUnavailable("Linked corpus or index paths are not supported.")
    return path


def sections(source, text):
    """Preserve headings; serialize aggregate table rows with repeated field names."""
    if source.endswith(".md"):
        heading, start, lines = "Introduction", 1, []
        for number, line in enumerate(text.splitlines(), 1):
            if re.match(r"^#{1,6}\s", line):
                if lines:
                    yield heading, f"lines {start}-{number - 1}", "\n".join(lines)
                heading, start, lines = line.lstrip("# ").strip(), number, []
            # Image URLs and artifact links are not followed or exposed as passages.
            line = re.sub(r"!\[[^]]*\]\([^)]*\)", "", line)
            line = re.sub(r"\[([^]]*)\]\([^)]*\)", r"\1", line)
            lines.append(line)
        if lines:
            yield heading, f"lines {start}-{len(text.splitlines())}", "\n".join(lines)
    elif source.endswith(".csv"):
        reader = csv.DictReader(StringIO(text))
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise CorpusUnavailable("Invalid aggregate table header.")
        if any(key.lower() in {"transactionid", "isfraud", "model_score", "row_number"} for key in reader.fieldnames):
            raise CorpusUnavailable("Row-level tables are not permitted.")
        for number, row in enumerate(reader, 2):
            if None in row or None in row.values():
                raise CorpusUnavailable("Malformed aggregate table.")
            yield "Aggregate results", f"CSV line {number}", "; ".join(f"{key}: {value}" for key, value in row.items())
    elif source.endswith(".sql"):
        yield "SQL methodology (text only; never executed)", f"lines 1-{len(text.splitlines())}", text
    elif source.endswith(".json"):
        value = json.loads(text)
        keys = SHAP_FIELDS if source.endswith("shap_summary.json") else value.keys()
        for key in keys:
            if key in value:
                yield key, f"JSON /{key}", f"{key}: {json.dumps(value[key], ensure_ascii=False, allow_nan=False)}"


def corpus(root):
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    documents, sources, skipped = [], [], []
    for entry in catalog["sources"]:
        source = entry["source"]
        if not re.fullmatch(r"reports/[a-z0-9_]+\.(md|csv|sql|json)", source):
            raise CorpusUnavailable("Invalid approved source catalog.")
        path = local_path(root, source)
        if not path.is_file():
            skipped.append({"source": source, "reason": "missing"})
            continue
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise CorpusUnavailable("An approved source changed. Review the source catalog before indexing.")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise CorpusUnavailable("An approved source changed. Review the source catalog before indexing.")
        text = content.decode("utf-8-sig")
        for section, locator, passage in sections(source, text):
            passage = passage.strip()
            if not passage:
                continue
            if SECRET.search(passage):
                raise CorpusUnavailable("Potential sensitive content detected; indexing refused.")
            documents.append({"text": passage, "source": source, "section": section,
                              "locator": locator, "source_sha256": entry["sha256"]})
        sources.append(entry)
    if not documents:
        raise CorpusUnavailable("No approved aggregate reports are available.")
    return documents, {"sources": sources, "skipped": skipped,
                       "corpus_sha256": fingerprint({"catalog": catalog, "sources": sources, "skipped": skipped})}
