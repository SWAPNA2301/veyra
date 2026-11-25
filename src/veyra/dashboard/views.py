"""Read-only analytics pages. These never import training or scoring routines."""

import pandas as pd
import streamlit as st

from .artifacts import nested
from . import charts
from .ui import chart, date_label, empty, header, insight, kpis, number, panel


def overview(store):
    quality = store.read("data_quality.json") or {}
    products = store.csv("01_product_risk.csv", ("product", "transactions", "fraud_transactions", "fraud_rate_pct"))
    trends = store.csv("03_time_trends.csv", ("elapsed_week", "transactions", "fraud_transactions", "fraud_rate_pct"))
    header("Portfolio intelligence", "Your fraud landscape, in focus.",
           "A clear view of observed transactions, changing fraud patterns, and where risk concentrates. Historical data, not a live feed.")
    total = nested(quality, "amount", "sum")
    kpis([
        ("Transactions", number(nested(quality, "tables", "transactions", "rows")), "Full IEEE-CIS dataset snapshot"),
        ("Fraud prevalence", number(nested(quality, "fraud", "fraud_rate_pct"), 2, "%"),
         f"{number(nested(quality, 'fraud', 'fraud_transactions'))} fraud-labeled transactions"),
        ("Transaction value", number(total / 1e6, 2, "M") if total is not None else "—", "Recorded units · currency unspecified"),
        ("Identity coverage", number(nested(quality, "join", "identity_coverage_pct"), 2, "%"),
         f"{number(nested(quality, 'join', 'matched_transactions'))} matched records"),
    ])
    left, right = st.columns([1.9, 1], gap="large")
    with left, st.container(border=True):
        panel("Fraud over time", "Transaction volume and fraud rate · relative weeks from TransactionDT")
        if trends is not None and not trends.empty:
            chart(charts.fraud_trends(trends), "overview_trends")
            st.caption("First and last bins may be partial weeks. Week numbers are not calendar dates.")
        else:
            empty("Trend report unavailable", "Restore the saved time-trend report to display this chart.")
    with right, st.container(border=True):
        panel("Where risk concentrates", "Fraud prevalence by product · all observed transactions")
        if products is not None and not products.empty:
            chart(charts.product_risk(products), "overview_product")
            highest = products.loc[products.fraud_rate_pct.idxmax()]
            st.caption(f"Product {highest['product']}: {number(highest['transactions'])} transactions support the highest observed rate.")
        else:
            empty("Product report unavailable", "Saved product aggregates will appear here when available.")
    first, second = st.columns(2, gap="large")
    with first:
        if quality.get("status") == "passed":
            insight("Validated foundations", "The saved audit passed its row, schema, label, amount and join-integrity checks.")
        else:
            insight("Audit not available", "Data quality cannot be inferred without the saved audit.", amber=True)
    with second:
        missing = nested(quality, "missing", "merged", "columns_over_50_pct_missing")
        insight("Missingness matters", f"{number(missing)} merged columns are more than 50% missing. Identity coverage is incomplete; absence is not a fraud label.", amber=True)
    with st.expander("Product breakdown & snapshot provenance"):
        st.caption(f"Saved audit: {date_label(quality.get('generated_at_utc'))}. Sources: data_quality.json, 01_product_risk.csv, 03_time_trends.csv.")
        if products is not None:
            table = products[["product", "transactions", "fraud_transactions", "fraud_rate_pct"]].rename(columns={
                "product": "Product", "transactions": "Transactions", "fraud_transactions": "Fraud labels", "fraud_rate_pct": "Fraud rate (%)"})
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "Transactions": st.column_config.NumberColumn(format="localized"),
                "Fraud labels": st.column_config.NumberColumn(format="localized"),
                "Fraud rate (%)": st.column_config.NumberColumn(format="%.2f"),
            })


def _baseline_table(store):
    frame = store.csv("training_metrics.csv", ("model", "pr_auc", "roc_auc", "precision", "recall", "f1", "threshold"))
    if frame is None:
        st.info("The saved training comparison is not available locally.")
        return
    table = frame[["model", "pr_auc", "roc_auc", "precision", "recall", "f1", "threshold"]].copy()
    table["model"] = table["model"].map({"no_skill": "No-skill prior", "logistic_regression": "Logistic Regression", "lightgbm": "LightGBM"}).fillna(table["model"])
    table.columns = ["Model", "PR-AUC (AP)", "ROC-AUC", "Precision", "Recall", "F1", "Threshold"]
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={name: st.column_config.NumberColumn(format="%.4f") for name in table.columns[1:]})
    st.caption("Saved validation model comparison at threshold 0.5. This is separate from the frozen-threshold holdout comparison above.")


def performance(store):
    holdout = store.read("holdout_evaluation.json")
    threshold = store.read("threshold_optimization.json")
    training = store.read("training_metrics.json")
    header("Model evaluation", "Performance, beyond accuracy.",
           "Validation and final holdout results at one frozen operating point. These are saved evaluations, not live reruns.")
    if not holdout or holdout.get("status") != "complete":
        empty("Final holdout report unavailable", "Restore a completed local holdout report. No evaluation will be started by this dashboard.")
        _baseline_table(store)
        return
    current_sha = nested(training, "models", "lightgbm", "sha256")
    if current_sha and nested(holdout, "fingerprints", "model_sha256") != current_sha:
        st.warning("The holdout report belongs to a different model run. Restore matching artifacts to compare results.")
        _baseline_table(store)
        return
    validation, test = holdout.get("validation"), holdout.get("test")
    needed = {"pr_auc", "roc_auc", "precision", "recall", "f1", "confusion_matrix", "rows"}
    if not all(isinstance(row, dict) and needed.issubset(row) for row in (validation, test)):
        st.warning("The saved holdout report is incomplete; metrics are not substituted.")
        return
    choice = st.segmented_control("Evaluation window", ["Final holdout", "Validation"], default="Final holdout")
    selected = validation if choice == "Validation" else test
    names = [("PR-AUC · average precision", "pr_auc", False), ("ROC-AUC", "roc_auc", False),
             ("Precision", "precision", True), ("Recall", "recall", True), ("F1 score", "f1", False)]
    cards = []
    for label, key, percent in names:
        delta = selected[key] - validation[key]
        detail = "Development estimate" if choice == "Validation" else f"{delta * (100 if percent else 1):+.2f}{' pp' if percent else ''} vs validation"
        cards.append((label, number(selected[key] * (100 if percent else 1), 2 if percent else 4, "%" if percent else ""), detail))
    kpis(cards)
    st.caption(f"LightGBM · frozen threshold {holdout['threshold']:.10f} · {number(selected['rows'])} rows in this view · {date_label(holdout.get('completed_at_utc'))}")
    left, right = st.columns([1.45, 1], gap="large")
    with left, st.container(border=True):
        panel("Validation → final holdout", "The same saved model and exact decision threshold")
        chart(charts.metric_comparison(validation, test), "performance_metrics")
    with right, st.container(border=True):
        panel("Decisions in context", f"{choice or 'Final holdout'} · actual rows, predicted columns")
        normalize = st.toggle("Normalize by actual class", value=True)
        chart(charts.confusion(selected, normalize), "performance_matrix")
    target = holdout.get("validation_recall_floor")
    if target is not None:
        insight("Frozen means frozen", f"The validation recall target was {target:.0%}; final test recall was {test['recall']:.2%}. The threshold was not adjusted after seeing the holdout.", amber=test["recall"] < target)
    with st.expander("Baseline comparison · validation at 0.5"):
        _baseline_table(store)
    show_curve = st.toggle("Explore the saved validation precision–recall curve", value=False)
    if show_curve:
        if threshold and threshold.get("model_sha256") == nested(holdout, "fingerprints", "model_sha256"):
            curve = store.csv("threshold_metrics.csv", ("recall", "precision", "flagged_rows"))
            if curve is not None and not curve.empty:
                chart(charts.precision_recall(curve, threshold["operating_points"], threshold["min_recall"]), "validation_pr_curve")
                st.caption("Only saved validation points are shown, sampled to at most 1,200 cutoffs for rendering. All metrics use the full saved calculations. No test curve or threshold tuning is performed.")
            else:
                st.info("Saved validation curve data is missing.")
        else:
            st.info("A matching validation threshold report is required for this curve.")
    st.caption("PR-AUC is average precision. Validation guided selection; test results were measured once. Probability calibration is not implied.")


def explainability(store):
    summary = store.read("shap_summary.json") or {}
    importance = store.csv("shap_global_importance.csv", ("feature", "mean_abs_grouped_shap"))
    header("Explainability", "Understand the signal.",
           "Existing SHAP explanations, grouped into original features. Read the model’s associations—not causal effects.")
    kpis([
        ("Validation sample", number(summary.get("sample_rows")), f"Reproducible seed {summary.get('seed', '—')}"),
        ("Original features", number(summary.get("original_features")), f"Grouped from {number(summary.get('encoded_columns'))} encoded columns"),
        ("Example model score", number(nested(summary, "local", "model_score"), 4), "Uncalibrated · one fixed validation case"),
    ])
    specs = [
        ("Global importance", "shap_global_importance.png", "Mean absolute grouped contributions", "Signed contributions are summed by original feature first, then their mean absolute magnitude is ranked."),
        ("SHAP distribution", "shap_beeswarm.png", "The spread behind the average", "Each dot is one sampled row. Positive contributions raise the model score; a single color avoids imposing an order on categories."),
        ("Local explanation", "shap_local_prediction.png", "One reproducible prediction", "This existing waterfall explains sample position 0. Contributions are raw log-odds, not probability percentage points."),
    ]
    for tab, (_, filename, title, detail) in zip(st.tabs([item[0] for item in specs]), specs):
        with tab:
            left, right = st.columns([2.4, 1], gap="large")
            with left, st.container(border=True):
                panel(title, "Saved figure · no SHAP computation is run by the dashboard")
                image = store.bytes(filename)
                if image is not None:
                    st.image(image, width="stretch", caption=title)
                    st.download_button("Download saved figure", image, file_name=filename, mime="image/png", key=f"download_{filename}")
                else:
                    empty("Saved figure unavailable", "Restore this local SHAP image; the dashboard will not recompute it.")
            with right:
                panel("How to read this", detail)
                if importance is not None and not importance.empty:
                    st.markdown("**Strongest global contributions**")
                    for index, row in importance.head(5).reset_index(drop=True).iterrows():
                        st.caption(f"{index+1:02d}  ·  {row['feature']}  —  {row['mean_abs_grouped_shap']:.3f}")
                insight("Preserved, not refitted", "These images use the saved pipeline and fitted preprocessing on a validation sample. The dashboard never accesses the raw dataset.")
                if summary.get("max_additivity_error") is not None:
                    st.caption(f"Saved reconstruction error: {summary['max_additivity_error']:.2e}.")
