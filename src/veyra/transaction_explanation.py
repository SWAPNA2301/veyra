"""On-demand TreeSHAP for submitted values only; no row cache or dataset reads."""

import csv
from io import StringIO
import math

from .batch_prediction import InvalidBatch, score_rows, validated_rows


class StalePrediction(ValueError):
    """The submitted batch provenance no longer matches local inference."""


def validate_values(values, schema):
    # Round-trip one record through the exact batch validator, including float32
    # bounds, blank handling and the feature allowlist. Never persist the CSV.
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(values.keys())
    writer.writerow(values.values())
    rows = validated_rows(stream.getvalue().encode("utf-8"), schema)
    if len(rows) != 1:
        raise InvalidBatch("Submit exactly one transaction for explanation.")
    return rows[0]


def explain_transaction(row, spec, *, expected_run, expected_threshold, expected_score):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from scipy.special import expit
    from .dashboard.predictor import load_fitted_pipeline
    from .explain import encoded_groups

    if expected_run != spec.run_id or expected_threshold != spec.threshold:
        raise StalePrediction
    # Reuse the batch scoring path, never replace its score with an explanation.
    scored = score_rows([row], spec)["rows"][0]
    if not math.isclose(scored["score"], expected_score, rel_tol=0, abs_tol=1e-12):
        raise StalePrediction
    pipeline = load_fitted_pipeline(spec)
    preprocess, classifier = pipeline.named_steps["preprocess"], pipeline.named_steps["classifier"]
    encoded = preprocess.transform(pd.DataFrame([row], columns=spec.selected))
    original, mapping = encoded_groups(preprocess, encoded.shape[1])
    iterations = classifier.best_iteration_ or None
    # Same native TreeSHAP and float64 accumulation as the existing SHAP report.
    contributions = classifier.booster_.predict(
        encoded.astype(np.float64), pred_contrib=True, num_iteration=iterations, num_threads=4)
    contributions = contributions.toarray() if sparse.issparse(contributions) else np.asarray(contributions)
    if contributions.shape != (1, encoded.shape[1] + 1) or not np.isfinite(contributions).all():
        raise ValueError("Invalid contribution output")
    base = float(contributions[0, -1])
    grouped = [float(contributions[0, :-1][mapping == name].sum()) for name in original]
    raw = float(classifier.booster_.predict(
        encoded, raw_score=True, num_iteration=iterations, num_threads=4)[0])
    error = abs(base + sum(grouped) - raw)
    if (not math.isfinite(raw) or not math.isclose(base + sum(grouped), raw, rel_tol=1e-7, abs_tol=1e-7)
            or not math.isclose(float(expit(raw)), scored["score"], rel_tol=1e-7, abs_tol=1e-7)):
        raise ValueError("Explanation does not reconstruct the score")
    return {
        "score": scored["score"], "flagged": scored["flagged"], "threshold": spec.threshold,
        "model_run_id": spec.run_id, "score_type": "uncalibrated",
        "base_value": base, "raw_margin": raw, "additivity_error": error,
        "contributions": [{"feature": name, "value": value,
                           "encoded_columns": int((mapping == name).sum())}
                          for name, value in zip(original, grouped)],
        "units": "raw_log_odds",
        "method": "LightGBM native TreeSHAP; tree-path-dependent training leaf counts",
        "grouping": "Signed sum of encoded contributions, including missing indicators, by original feature; not recomputed coalition Shapley values.",
        "notice": "Model contributions, not causal explanations. Scores are uncalibrated; flags are predictions, not confirmed fraud.",
    }
