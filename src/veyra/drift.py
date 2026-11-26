"""Fixed, train-only distribution profiles and stateless batch drift checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import ALLOWED_FEATURES, NUMERIC_FEATURES

PROFILE_FILE = "drift_reference.json"
POLICY = {
    "min_batch_rows": 200, "min_observed_rows": 100, "min_reference_rows": 1000,
    "numeric_quantile_bins": 10, "max_categories": 20, "min_category_count": 50,
    "epsilon": 1e-6, "psi_watch": 0.10, "psi_high": 0.25,
    "missing_watch_pp": 5.0, "missing_high_pp": 10.0,
}


class ReferenceUnavailable(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def normalized(series, numeric):
    if numeric:
        result = pd.to_numeric(series, errors="raise").astype(float)
        if not np.isfinite(result.dropna()).all():
            raise ValueError("Nonfinite reference values")
        return result
    return series.astype("string").str.strip().replace("", pd.NA)


def bin_counts(series, feature):
    missing = series.isna()
    observed = series[~missing]
    if feature["kind"] == "numeric":
        cuts = feature["cuts"]
        indices = np.searchsorted(cuts, observed.to_numpy(), side="right") + 1
        if feature["minimum"] is not None:
            indices[observed.to_numpy() < feature["minimum"]] = 0
            indices[observed.to_numpy() > feature["maximum"]] = len(cuts) + 2
        counts = np.bincount(indices, minlength=len(cuts) + 3).tolist()
    else:
        counts_by_value = observed.value_counts()
        counts = [int(counts_by_value.get(value, 0)) for value in feature["categories"]]
        counts.append(len(observed) - sum(counts))
    return [*counts, int(missing.sum())]


def feature_profile(name, series):
    numeric = name in NUMERIC_FEATURES
    series = normalized(series, numeric)
    observed = series.dropna()
    feature = {"name": name, "kind": "numeric" if numeric else "categorical"}
    if numeric:
        lo, hi = (float(observed.min()), float(observed.max())) if len(observed) else (None, None)
        cuts = sorted(set(float(x) for x in observed.quantile(np.arange(1, 10) / 10))) if len(observed) else []
        cuts = [x for x in cuts if lo < x < hi]
        feature.update(minimum=lo, maximum=hi, cuts=cuts)
        if lo is None:
            labels = ["Below reference range", "No observed reference values", "Above reference range"]
        else:
            boundaries = [lo, *cuts, hi]
            labels = [f"< {lo:.6g}", *[f"[{a:.6g}, {b:.6g}{']' if i == len(boundaries)-2 else ')'}"
                      for i, (a, b) in enumerate(zip(boundaries, boundaries[1:]))], f"> {hi:.6g}"]
    else:
        counts = observed.value_counts()
        categories = sorted((str(value) for value in counts.index if counts[value] >= POLICY["min_category_count"]),
                            key=lambda value: (-int(counts[value]), value))[:POLICY["max_categories"]]
        feature["categories"] = categories
        labels = [*(f"Category: {value}" for value in categories), "Other / unseen"]
    feature["labels"] = [*labels, "Missing"]
    feature["counts"] = bin_counts(series, feature)
    return feature


def load_reference(root: Path):
    """Read only the aggregate profile; never build/update it in an HTTP request."""
    try:
        profile = json.loads((root / "reports" / PROFILE_FILE).read_text(encoding="utf-8"))
        payload = {key: value for key, value in profile.items() if key != "profile_id"}
        training_bytes = (root / "reports" / "training_metrics.json").read_bytes()
        training = json.loads(training_bytes)
        split_bytes = (root / "reports" / "chronological_split_summary.json").read_bytes()
        if (profile["profile_id"] != digest(payload) or profile["schema_version"] != 1
                or profile["policy"] != POLICY or profile["source_split"] != "train"
                or profile["reference_rows"] < POLICY["min_reference_rows"]
                or profile["training_metrics_sha256"] != hashlib.sha256(training_bytes).hexdigest()
                or profile["split_audit_sha256"] != hashlib.sha256(split_bytes).hexdigest()
                or profile["model_run_id"] != training["run_id"]
                or [item["name"] for item in profile["features"]] != training["features"]["selected"]):
            raise ValueError
        for item in profile["features"]:
            if item["name"] not in ALLOWED_FEATURES or len(item["counts"]) != len(item["labels"]):
                raise ValueError
            if any(type(count) is not int or count < 0 for count in item["counts"]) or sum(item["counts"]) != profile["reference_rows"]:
                raise ValueError
            if item["kind"] == "numeric":
                if (item["name"] not in NUMERIC_FEATURES or len(item["counts"]) != len(item["cuts"]) + 4
                        or item["cuts"] != sorted(set(item["cuts"])) or not np.isfinite(item["cuts"]).all()):
                    raise ValueError
            elif item["kind"] != "categorical" or len(item["counts"]) != len(item["categories"]) + 2:
                raise ValueError
        return profile
    except Exception:
        raise ReferenceUnavailable("A matching fixed training reference is unavailable. Create it once with the manual reference command; existing references are never overwritten automatically.") from None


def reference_summary(profile):
    return {key: profile[key] for key in ("profile_id", "created_at_utc", "source_split", "reference_rows",
                                          "model_run_id", "train_window", "policy", "method")}


def psi(reference, current):
    p, q = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    p = p / p.sum() + POLICY["epsilon"]
    q = q / q.sum() + POLICY["epsilon"]
    p /= p.sum()
    q /= q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def severity(value, watch, high):
    return "high" if value >= high else "watch" if value >= watch else "stable"


def compare_rows(rows, provided, profile):
    """Return aggregate counts only. Neither inputs nor results enter a cache."""
    total, ref_total = len(rows), profile["reference_rows"]
    enough = total >= POLICY["min_batch_rows"]
    frame = pd.DataFrame(rows)
    features = []
    for reference in profile["features"]:
        name = reference["name"]
        values = normalized(frame[name], reference["kind"] == "numeric")
        counts = bin_counts(values, reference)
        ref_counts = reference["counts"]
        observed, ref_observed = total - counts[-1], ref_total - ref_counts[-1]
        eligible = enough and min(observed, ref_observed) >= POLICY["min_observed_rows"]
        value = psi(ref_counts, counts) if eligible else None
        missing_delta = 100 * (counts[-1] / total - ref_counts[-1] / ref_total)
        psi_level = severity(value, POLICY["psi_watch"], POLICY["psi_high"]) if eligible else None
        missing_level = severity(abs(missing_delta), POLICY["missing_watch_pp"], POLICY["missing_high_pp"]) if enough else None
        combined = max((level for level in (psi_level, missing_level) if level),
                       key=lambda level: ("stable", "watch", "high").index(level), default=None)
        features.append({
            "name": name, "kind": reference["kind"], "provided": name in provided,
            "psi": value, "psi_severity": psi_level, "missing_severity": missing_level, "severity": combined,
            "psi_status": "ready" if eligible else "insufficient_data",
            "observed_rows": observed, "reference_observed_rows": ref_observed,
            "missing_rate": counts[-1] / total, "reference_missing_rate": ref_counts[-1] / ref_total,
            "missing_delta_pp": missing_delta,
            "bins": [{"label": label, "reference_count": rc, "batch_count": bc,
                      "reference_share": rc / ref_total, "batch_share": bc / total}
                     for label, rc, bc in zip(reference["labels"], ref_counts, counts)],
        })
    ranked = sorted(features, key=lambda f: (-( {None: -1, "stable": 0, "watch": 1, "high": 2}[f["severity"]]),
                                            -(f["psi"] if f["psi"] is not None else -1), -abs(f["missing_delta_pp"]), f["name"]))
    assessed = [f for f in features if f["psi"] is not None]
    return {
        "status": "ready" if enough else "insufficient_data", "rows": total,
        "reference": reference_summary(profile), "features": ranked,
        "summary": {"features": len(features), "provided_features": len(provided), "psi_assessed": len(assessed),
                    "high": sum(f["severity"] == "high" for f in features) if enough else None,
                    "watch": sum(f["severity"] == "watch" for f in features) if enough else None,
                    "max_psi": max((f["psi"] for f in assessed), default=None),
                    "max_missing_change_pp": max(abs(f["missing_delta_pp"]) for f in features) if enough else None},
        "notice": "Distribution drift is not evidence of fraud, performance loss or concept drift. Severity thresholds are fixed review heuristics, not significance tests. No automatic action or reference update.",
    }
