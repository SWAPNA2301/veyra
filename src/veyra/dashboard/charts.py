"""Plotly views of existing aggregate metrics, with shared dark styling."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MINT, BLUE, AMBER = "#53deb5", "#78b8f4", "#e8b577"


def style(figure, *, height=340):
    figure.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(family="Segoe UI, sans-serif", size=12, color="#92aeb9"),
                         margin=dict(l=12, r=18, t=25, b=32),
                         legend=dict(orientation="h", y=1.16, x=0, font=dict(size=10)),
                         hoverlabel=dict(bgcolor="#16303a", bordercolor="#35565e", font_color="#eff8f5"))
    figure.update_xaxes(showgrid=False, zeroline=False, automargin=True)
    figure.update_yaxes(gridcolor="#1d333d", zeroline=False, automargin=True)
    return figure


def fraud_trends(frame):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=frame.elapsed_week, y=frame.transactions, name="Transactions", marker_color="#20454f",
                         hovertemplate="Week %{x}<br>%{y:,} transactions<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=frame.elapsed_week, y=frame.fraud_rate_pct, name="Fraud rate",
                             mode="lines+markers", line=dict(color=MINT, width=2.8), marker=dict(size=5),
                             customdata=frame.fraud_transactions,
                             hovertemplate="Week %{x}<br>Fraud rate %{y:.2f}%<br>%{customdata:,.0f} fraud labels<extra></extra>"), secondary_y=True)
    style(fig, height=345)
    fig.update_layout(bargap=0.35, hovermode="x unified")
    fig.update_xaxes(title_text="Relative week")
    fig.update_yaxes(title_text="Transactions", secondary_y=False)
    fig.update_yaxes(title_text="Fraud rate (%)", secondary_y=True, showgrid=False, rangemode="tozero")
    return fig


def product_risk(frame):
    data = frame.sort_values("fraud_rate_pct")
    fig = go.Figure(go.Bar(x=data.fraud_rate_pct, y=data["product"], orientation="h",
                          marker_color=[AMBER if value == data.fraud_rate_pct.max() else MINT for value in data.fraud_rate_pct],
                          text=[f"{value:.2f}%" for value in data.fraud_rate_pct], textposition="outside",
                          customdata=data[["transactions", "fraud_transactions"]].to_numpy(),
                          hovertemplate="Product %{y}<br>Fraud rate %{x:.2f}%<br>%{customdata[0]:,.0f} transactions<br>%{customdata[1]:,.0f} fraud labels<extra></extra>"))
    style(fig, height=345)
    fig.update_xaxes(title_text="Fraud rate (%)", range=[0, max(1, data.fraud_rate_pct.max() * 1.3)])
    fig.update_layout(showlegend=False, bargap=0.48)
    return fig


def metric_comparison(validation, test):
    names = ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"]
    keys = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
    fig = go.Figure()
    for label, data, color in (("Validation", validation, BLUE), ("Test", test, MINT)):
        fig.add_trace(go.Bar(name=label, x=names, y=[data[key] for key in keys], marker_color=color,
                             text=[f"{data[key]:.3f}" for key in keys], textposition="outside",
                             hovertemplate=label + "<br>%{x}: %{y:.4f}<extra></extra>"))
    style(fig)
    fig.update_layout(barmode="group", bargap=0.36, bargroupgap=0.12)
    fig.update_yaxes(range=[0, 1.05], title_text="Metric value")
    return fig


def confusion(data, normalize=True):
    counts = np.asarray(data["confusion_matrix"], dtype=int)
    ratios = counts / counts.sum(axis=1, keepdims=True)
    labels = ["Non-fraud", "Fraud"]
    fig = go.Figure(go.Heatmap(z=ratios if normalize else counts, x=labels, y=labels,
                              colorscale=[[0, "#102630"], [1, MINT]], showscale=False, xgap=6, ygap=6,
                              customdata=counts, zmin=0, zmax=1 if normalize else int(counts.max()),
                              hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{customdata:,} transactions<extra></extra>"))
    for actual in range(2):
        for predicted in range(2):
            fraction = ratios[actual, predicted] if normalize else counts[actual, predicted] / counts.max()
            fig.add_annotation(x=labels[predicted], y=labels[actual], showarrow=False,
                               text=f"<b>{counts[actual,predicted]:,}</b><br>{ratios[actual,predicted]:.1%} of class",
                               font=dict(size=14, color="#062a22" if fraction > .55 else "#d8e9ed"))
    style(fig)
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="Actual", autorange="reversed", showgrid=False)
    return fig


def precision_recall(sweep, points, floor):
    # Render real saved points only; model metrics remain the full saved values.
    shown = sweep.loc[sweep.flagged_rows > 0].iloc[np.linspace(0, max(0, (sweep.flagged_rows > 0).sum()-1),
                                                            min(1200, (sweep.flagged_rows > 0).sum()), dtype=int)]
    fig = go.Figure(go.Scatter(x=shown.recall, y=shown.precision, mode="lines", name="Validation curve",
                             line=dict(color=BLUE, width=2), hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>"))
    for point, color in zip(points, ("#8ea4ae", MINT, AMBER)):
        fig.add_trace(go.Scatter(x=[point["recall"]], y=[point["precision"]], mode="markers",
                                 name=point["policy"].replace("_", " "), marker=dict(size=11, color=color),
                                 hovertemplate=f"Threshold {point['threshold']:.6f}<br>Recall %{{x:.3f}}<br>Precision %{{y:.3f}}<extra></extra>"))
    fig.add_vline(x=floor, line_dash="dot", line_color="#607e88")
    style(fig, height=370)
    fig.update_xaxes(title_text="Recall", range=[0, 1])
    fig.update_yaxes(title_text="Precision", range=[0, 1])
    return fig


def score_gauge(score, threshold):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score,
        number=dict(valueformat=".4f", font=dict(size=43, color="#eaf7f3")),
        gauge=dict(axis=dict(range=[0, 1], tickvals=[0, .5, 1], tickcolor="#50717d"),
                   bar=dict(color=AMBER if score >= threshold else MINT, thickness=.22),
                   bgcolor="#132932", borderwidth=0,
                   steps=[dict(range=[0, threshold], color="#183c34"), dict(range=[threshold, 1], color="#443625")],
                   threshold=dict(line=dict(color="#eaf7f3", width=3), thickness=.75, value=threshold))))
    style(fig, height=240)
    fig.update_layout(margin=dict(l=20, r=20, t=24, b=15))
    return fig
