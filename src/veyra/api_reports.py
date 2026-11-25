"""Read-only, allowlisted aggregate reports for the local React dashboard."""

import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path


IMAGES = {
    "global": "shap_global_importance.png",
    "beeswarm": "shap_beeswarm.png",
    "local": "shap_local_prediction.png",
}
REPORTS = {
    "data_quality.json", "training_metrics.json", "threshold_optimization.json",
    "holdout_evaluation.json", "shap_summary.json", "split_audit.json",
    "01_product_risk.csv", "03_time_trends.csv", "training_metrics.csv",
    "shap_global_importance.csv", "threshold_metrics.csv", *IMAGES.values(),
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def dig(value, *keys):
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


@lru_cache(maxsize=32)
def _read(path: str, modified: int, size: int):
    # The signature invalidates cached content when a local report changes.
    file = Path(path)
    if file.suffix == ".json":
        result = json.loads(file.read_text(encoding="utf-8-sig"),
                            parse_constant=lambda _: None)
        return result if isinstance(result, dict) else {}
    with file.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ReportStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.reports = self.root / "reports"
        self.warnings: list[str] = []

    def path(self, name: str) -> Path:
        if name not in REPORTS:
            raise ValueError("Unsupported report")
        path = (self.reports / name).resolve()
        if not path.is_relative_to(self.reports.resolve()):
            raise ValueError("Report path escapes reports directory")
        return path

    def read(self, name: str):
        try:
            path = self.path(name)
            stat = path.stat()
            if stat.st_size > 20_000_000:
                raise ValueError("Oversized aggregate report")
            return _read(str(path), stat.st_mtime_ns, stat.st_size)
        except (OSError, ValueError, csv.Error, UnicodeError):
            self.warnings.append(f"{name} is missing or unreadable; its view is unavailable.")
            return {} if name.endswith(".json") else []

    def rows(self, name: str, numeric: tuple[str, ...], text: tuple[str, ...] = ()):
        result = []
        try:
            for row in self.read(name):
                values = {key: float(row[key]) for key in numeric}
                if not all(finite(value) for value in values.values()):
                    raise ValueError("Non-finite aggregate")
                result.append({**values, **{key: row[key] for key in text}})
        except (KeyError, TypeError, ValueError):
            self.warnings.append(f"{name} has an invalid aggregate schema; its view is unavailable.")
            return []
        return result

    def digest(self, name: str):
        try:
            return hashlib.sha256(self.path(name).read_bytes()).hexdigest()
        except OSError:
            return None


def manifests(store):
    training = store.read("training_metrics.json")
    frozen = store.read("threshold_optimization.json")
    points = frozen.get("operating_points", [])
    points = [p for p in points if isinstance(p, dict)] if isinstance(points, list) else []
    chosen = [p for p in points if p.get("policy") == "max_precision_at_recall"]
    threshold = chosen[0].get("threshold") if len(chosen) == 1 else None
    sha = dig(training, "models", "lightgbm", "sha256")
    valid = bool(sha and finite(threshold) and 0 <= threshold <= 1
                 and frozen.get("model_sha256") == sha
                 and frozen.get("model_run_id") == training.get("run_id")
                 and frozen.get("training_metrics_sha256") == store.digest("training_metrics.json")
                 and frozen.get("selection_split") == "validation"
                 and training.get("preprocessing_fit_split") == "train")
    if not valid:
        store.warnings.append("Matching training and frozen validation manifests are unavailable.")
    return training, frozen, threshold if valid else None, valid


def metric_block(value):
    if not isinstance(value, dict):
        return None
    keys = ("pr_auc", "roc_auc", "precision", "recall", "f1", "rows", "fraud_rows")
    matrix = value.get("confusion_matrix")
    if not (all(finite(value.get(k)) for k in keys) and isinstance(matrix, list)
            and len(matrix) == 2 and all(isinstance(row, list) and len(row) == 2
            and all(finite(n) and n >= 0 for n in row) for row in matrix)):
        return None
    return {**{k: value[k] for k in keys}, "confusion_matrix": matrix}


def shap_manifest(store, training):
    shap = store.read("shap_summary.json")
    valid = bool(shap and shap.get("validation_only") is True
                 and shap.get("model_sha256") == dig(training, "models", "lightgbm", "sha256")
                 and shap.get("model_run_id") == training.get("run_id"))
    if not valid:
        store.warnings.append("Matching validation-only SHAP artifacts are unavailable.")
    return shap if valid else {}


def dashboard(root: Path):
    store = ReportStore(root)
    quality = store.read("data_quality.json")
    training, frozen, threshold, matching = manifests(store)
    holdout = store.read("holdout_evaluation.json")
    if not (matching and holdout.get("status") == "complete"
            and holdout.get("model_run_id") == training.get("run_id")
            and dig(holdout, "fingerprints", "model_sha256") == dig(training, "models", "lightgbm", "sha256")
            and dig(holdout, "fingerprints", "threshold_audit_sha256") == store.digest("threshold_optimization.json")
            and holdout.get("threshold") == threshold):
        holdout = {}
        store.warnings.append("A matching completed holdout report is unavailable. No evaluation was run.")
    shap = shap_manifest(store, training)
    model = {"available": False, "reason": None, "threshold": threshold,
             "sha256": dig(training, "models", "lightgbm", "sha256"),
             "run_id": training.get("run_id"), "features": []}
    try:
        from .dashboard.predictor import model_spec
        spec = model_spec(store)
        model.update(available=True, features=list(spec.selected))
    except (ImportError, OSError, ValueError):
        model["reason"] = "Matching local model, manifests, or inference dependencies are unavailable. Report pages remain accessible."
    products = store.rows("01_product_risk.csv", ("transactions", "fraud_transactions", "fraud_rate_pct", "total_amount", "fraud_amount", "fraud_amount_share_pct"), ("product",))
    trends = store.rows("03_time_trends.csv", ("elapsed_week", "transactions", "fraud_transactions", "fraud_rate_pct", "mean_amount", "identity_coverage_pct"))
    baselines = store.rows("training_metrics.csv", ("pr_auc", "roc_auc", "precision", "recall", "f1", "threshold"), ("model",)) if training else []
    importance = store.rows("shap_global_importance.csv", ("mean_abs_grouped_shap",), ("feature",)) if shap else []
    return {
        "overview": {
            "status": quality.get("status"), "generated_at": quality.get("generated_at_utc"),
            "transactions": dig(quality, "tables", "transactions", "rows"),
            "fraud_transactions": dig(quality, "fraud", "fraud_transactions"),
            "fraud_rate_pct": dig(quality, "fraud", "fraud_rate_pct"),
            "transaction_value": dig(quality, "amount", "sum"),
            "identity_coverage_pct": dig(quality, "join", "identity_coverage_pct"),
            "identity_matches": dig(quality, "join", "matched_transactions"),
            "missing_columns": dig(quality, "missing", "merged", "columns_over_50_pct_missing"),
            "duplicate_rows": dig(quality, "tables", "transactions", "exact_duplicate_rows"),
            "products": products, "trends": trends,
        },
        "performance": {
            "validation": metric_block(holdout.get("validation")), "test": metric_block(holdout.get("test")),
            "threshold": threshold, "recall_floor": frozen.get("min_recall") if matching else None,
            "completed_at": holdout.get("completed_at_utc"), "baselines": baselines,
            "operating_points": frozen.get("operating_points", []) if matching else [],
        },
        "explainability": {
            **{key: shap.get(key) for key in ("sample_rows", "seed", "original_features", "encoded_columns", "max_additivity_error")},
            "local_score": dig(shap, "local", "model_score"), "importance": importance,
            "images": [{"key": key, "url": f"/api/explanations/{key}",
                        "available": bool(shap) and store.path(name).is_file()} for key, name in IMAGES.items()],
        },
        "model": model, "warnings": list(dict.fromkeys(store.warnings)),
    }


def precision_recall(root: Path):
    store = ReportStore(root)
    _, frozen, _, valid = manifests(store)
    rows = store.rows("threshold_metrics.csv", ("precision", "recall", "threshold")) if valid else []
    if not rows:
        raise ValueError("The matching saved validation curve is unavailable.")
    if len(rows) > 1200:
        rows = [rows[round(i * (len(rows) - 1) / 1199)] for i in range(1200)]
    return {"points": rows, "operating_points": frozen.get("operating_points", [])}
