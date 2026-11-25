"""Single-row manual inference. ML libraries and the model load only on submit."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path

import streamlit as st

from . import charts
from .ui import chart, empty, header, insight, number, panel


class ScoringUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    root: Path
    path: Path
    sha256: str
    threshold: float
    run_id: str
    selected: tuple[str, ...]
    source_hashes: tuple[tuple[str, str], ...]


def model_spec(store) -> ModelSpec:
    training, frozen = store.read("training_metrics.json"), store.read("threshold_optimization.json")
    if not training or not frozen:
        raise ScoringUnavailable("Matching training metrics and a frozen validation threshold are required for scoring.")
    try:
        saved = training["models"]["lightgbm"]
        points = [item for item in frozen["operating_points"] if item["policy"] == "max_precision_at_recall"]
        threshold = float(points[0]["threshold"]) if len(points) == 1 else float("nan")
        metrics_hash = hashlib.sha256((store.reports / "training_metrics.json").read_bytes()).hexdigest()
        if (frozen["model_sha256"] != saved["sha256"] or frozen["model_run_id"] != training["run_id"]
                or frozen["training_metrics_sha256"] != metrics_hash or frozen.get("selection_split") != "validation"
                or training.get("preprocessing_fit_split") != "train" or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ScoringUnavailable("Model and frozen-threshold provenance do not match. Scoring is disabled.")
        path = (store.reports / saved["artifact"]).resolve()
        if not path.is_relative_to((store.root / "models").resolve()) or path.suffix != ".joblib":
            raise ScoringUnavailable("The model must be a trusted local joblib artifact inside this project's models folder.")
        if not path.is_file():
            raise ScoringUnavailable("The saved LightGBM pipeline is missing locally. Restore the matching file in models; all report pages remain available.")
        return ModelSpec(store.root, path, saved["sha256"], threshold, training["run_id"],
                         tuple(training["features"]["selected"]), tuple(sorted(training["source_code_sha256"].items())))
    except ScoringUnavailable:
        raise
    except (KeyError, TypeError, OSError, AttributeError, ValueError) as exc:
        raise ScoringUnavailable("The local scoring manifest is incomplete or unreadable.") from exc


@st.cache_resource(show_spinner=False, max_entries=2)
def _load_pipeline(path: str, expected_hash: str, root: str, source_hashes: tuple, signature: tuple):
    # Heavy imports intentionally kept out of page startup/report views.
    import joblib
    import lightgbm as lgb
    import numpy as np
    from sklearn.pipeline import Pipeline

    file = Path(path)
    if hashlib.sha256(file.read_bytes()).hexdigest() != expected_hash:
        raise ScoringUnavailable("Saved model checksum mismatch. Restore the original matching pipeline.")
    for name, expected in source_hashes:
        if name not in {"features.py", "training.py", "training_data.py"}:
            raise ScoringUnavailable("Unrecognized model source manifest.")
        source = Path(root) / "src" / "veyra" / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ScoringUnavailable("The fitted model's source files no longer match. Scoring is disabled.")
    model = joblib.load(file)
    if not isinstance(model, Pipeline) or not isinstance(model.named_steps.get("classifier"), lgb.LGBMClassifier):
        raise ScoringUnavailable("The local artifact is not the expected LightGBM pipeline.")
    if not np.array_equal(model.classes_, [0, 1]):
        raise ScoringUnavailable("Unsupported classifier output classes.")
    return model


def score_manual(values: dict, spec: ModelSpec) -> dict:
    import pandas as pd

    for name in ("TransactionAmt", "dist1", "dist2"):
        value = values.get(name)
        if (name == "TransactionAmt" and value is None) or (value is not None and (not math.isfinite(value) or value < 0)):
            raise ScoringUnavailable("Enter a finite, nonnegative transaction amount and valid optional distances.")
    status = spec.path.stat()
    source_stamps = tuple((name, (spec.root / "src" / "veyra" / name).stat().st_mtime_ns) for name, _ in spec.source_hashes)
    pipeline = _load_pipeline(str(spec.path), spec.sha256, str(spec.root), spec.source_hashes,
                              (status.st_mtime_ns, status.st_size, source_stamps))
    fitted = tuple(pipeline.named_steps["preprocess"].named_steps["policy"].selected_columns_)
    if fitted != spec.selected:
        raise ScoringUnavailable("The saved feature manifest and pipeline schema disagree.")
    row = {name: values.get(name) for name in fitted}
    probability = float(pipeline.predict_proba(pd.DataFrame([row]))[0, 1])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ScoringUnavailable("The model returned an invalid score.")
    return {"score": probability, "threshold": spec.threshold, "above": probability >= spec.threshold,
            "provided": sum(value is not None for value in row.values()), "expected": len(row),
            "amount": row["TransactionAmt"], "submitted_at": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            "run_id": spec.run_id, "model_sha": spec.sha256}


def predictor(store):
    header("Manual inference", "Inspect a transaction.",
           "Explore one manually entered scenario with the saved LightGBM pipeline. Inputs and results stay in this session; nothing is added to the dataset.",
           badge="LOCAL INFERENCE ONLY")
    try:
        spec, unavailable = model_spec(store), None
    except (ScoringUnavailable, ValueError) as exc:
        spec, unavailable = None, str(exc)
    if unavailable:
        st.warning(unavailable)
    else:
        st.caption(f"Local artifact found · loaded only on submit · frozen threshold {spec.threshold:.10f}")
    left, right = st.columns([1.8, 1], gap="large")
    values = {}
    with left:
        with st.form("manual_transaction", border=True):
            panel("Transaction details", "Amount is required. Leave unknown fields empty; the saved preprocessing handles missing values.")
            c1, c2 = st.columns(2)
            values["TransactionAmt"] = c1.number_input("Transaction amount · recorded units", min_value=0.0, value=None, step=1.0, format="%.3f", placeholder="Enter an amount")
            products = store.csv("01_product_risk.csv", ("product",))
            if products is not None:
                values["ProductCD"] = c2.selectbox("Product code", sorted(products["product"].dropna().astype(str).unique()), index=None, placeholder="Unknown")
            else:
                values["ProductCD"] = c2.text_input("Product code", placeholder="Unknown").strip() or None
            values["card4"] = c1.selectbox("Card network", ["visa", "mastercard", "american express", "discover"], index=None, placeholder="Unknown", format_func=str.title)
            values["card6"] = c2.selectbox("Card type", ["credit", "debit", "charge card", "debit or credit"], index=None, placeholder="Unknown", format_func=str.title)
            values["P_emaildomain"] = c1.text_input("Purchaser email domain", placeholder="Unknown", help="Domain only, not a full email address.").strip() or None
            values["R_emaildomain"] = c2.text_input("Recipient email domain", placeholder="Unknown").strip() or None
            with st.expander("Device context · optional"):
                a, b = st.columns(2)
                values["DeviceType"] = a.selectbox("Device type", ["desktop", "mobile"], index=None, placeholder="Unknown", format_func=str.title)
                values["DeviceInfo"] = b.text_input("Device information", placeholder="Unknown").strip() or None
                values["id_30"] = a.text_input("Operating system", placeholder="Unknown").strip() or None
                values["id_31"] = b.text_input("Browser", placeholder="Unknown").strip() or None
                values["id_33"] = a.text_input("Screen resolution", placeholder="Unknown").strip() or None
            with st.expander("Distance & match signals · optional"):
                a, b = st.columns(2)
                values["dist1"] = a.number_input("Distance signal 1 · dataset units", min_value=0.0, value=None, placeholder="Unknown")
                values["dist2"] = b.number_input("Distance signal 2 · dataset units", min_value=0.0, value=None, placeholder="Unknown")
                st.caption("Use original dataset signal codes. Do not guess unknown attributes.")
                cols = st.columns(3)
                for index in range(1, 10):
                    options = ["M0", "M1", "M2"] if index == 4 else ["T", "F"]
                    values[f"M{index}"] = cols[(index-1) % 3].selectbox(f"Match signal M{index}", options, index=None, placeholder="Unknown")
            submitted = st.form_submit_button("Analyze transaction", type="primary", icon=":material/arrow_forward:", width="stretch", disabled=spec is None)
        if submitted and spec:
            st.session_state.pop("risk_result", None)
            try:
                with st.spinner("Loading the saved pipeline and scoring this scenario…"):
                    st.session_state["risk_result"] = score_manual(values, spec)
            except (ScoringUnavailable, OSError, ImportError, ValueError) as exc:
                st.error(f"Scoring unavailable: {exc}")
            except Exception:
                st.error("The local model could not score this input. Confirm matching artifacts and installed dependencies; report pages remain available.")
    with right, st.container(border=True):
        panel("Risk assessment", "Last submitted scenario · not a verified fraud decision")
        result = st.session_state.get("risk_result")
        if (result and spec and result["run_id"] == spec.run_id
                and result.get("model_sha") == spec.sha256 and result["threshold"] == spec.threshold):
            chart(charts.score_gauge(result["score"], result["threshold"]), "manual_risk_score")
            st.caption("Uncalibrated model score · 0–1 scale, not a fraud probability")
            insight("Above frozen threshold · review suggested" if result["above"] else "Below frozen threshold",
                    f"Score {result['score']:.6f} vs threshold {result['threshold']:.6f}. This threshold is fixed; it is not adjusted from manual inputs.", amber=result["above"])
            st.caption(f"Amount: {number(result['amount'], 3)} recorded units · {result['provided']}/{result['expected']} fields provided.")
            st.caption(f"Submitted {result['submitted_at']}. Submit again after changing inputs.")
            if st.button("Clear result", width="stretch"):
                st.session_state.pop("risk_result", None)
                st.rerun()
        else:
            empty("Awaiting your input", "Enter a transaction amount and any known context, then analyze. No score is generated until you submit.")
        st.markdown("**A score, not certainty.**")
        st.caption("The model was class-weighted and is not probability-calibrated. Missing inputs use the training-fitted rules; unseen categories use the existing encoder. No automatic transaction action is taken.")
