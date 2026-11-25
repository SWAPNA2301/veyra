"""Plotly views of existing aggregate metrics, with shared dark styling."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MINT, BLUE, AMBER = "#67e8c2", "#91b5fb", "#e8bb7d"


def style(figure, *, height=320):
    figure.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(family="Inter, Segoe UI, sans-serif", size=11, color="#acbdd1"),
                         margin=dict(l=8, r=20, t=35, b=22),
                         legend=dict(orientation="h", y=1.13, x=0, font=dict(size=11), traceorder="normal"),
                         hoverlabel=dict(bgcolor="#202e43", bordercolor="#42556e", font_color="#f1f6fc"))
    figure.update_xaxes(showgrid=False, zeroline=False, automargin=True, tickfont=dict(size=10), title_font=dict(size=11))
    figure.update_yaxes(gridcolor="#263347", zeroline=False, automargin=True, tickfont=dict(size=10), title_font=dict(size=11))
    return figure


def fraud_trends(frame):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=frame.elapsed_week, y=frame.transactions, name="Transactions", marker_color="#304a67", marker_cornerradius=3,
                         hovertemplate="Week %{x}<br>%{y:,} transactions<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=frame.elapsed_week, y=frame.fraud_rate_pct, name="Fraud rate",
                             mode="lines+markers", line=dict(color=MINT, width=2.8), marker=dict(size=5),
                             customdata=frame.fraud_transactions,
                             hovertemplate="Week %{x}<br>Fraud rate %{y:.2f}%<br>%{customdata:,.0f} fraud labels<extra></extra>"), secondary_y=True)
    style(fig, height=310)
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
                          textfont=dict(color="#e3ecf8", size=11), marker_cornerradius=4, cliponaxis=False,
                          customdata=data[["transactions", "fraud_transactions"]].to_numpy(),
                          hovertemplate="Product %{y}<br>Fraud rate %{x:.2f}%<br>%{customdata[0]:,.0f} transactions<br>%{customdata[1]:,.0f} fraud labels<extra></extra>"))
    style(fig, height=310)
    fig.update_xaxes(title_text="Fraud rate (%)", range=[0, max(1, data.fraud_rate_pct.max() * 1.3)])
    fig.update_layout(showlegend=False, bargap=0.48)
    return fig


def metric_comparison(validation, test):
    names = ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"]
    keys = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
    fig = go.Figure()
    for label, data, color in (("Validation", validation, BLUE), ("Test", test, MINT)):
        fig.add_trace(go.Bar(name=label, y=names, x=[data[key] for key in keys], orientation="h", marker_color=color,
                             marker_cornerradius=3, cliponaxis=False, textfont=dict(color="#dae5f5", size=11),
                             text=[f"{data[key]:.3f}" for key in keys], textposition="outside",
                             hovertemplate=label + "<br>%{y}: %{x:.4f}<extra></extra>"))
    style(fig, height=365)
    fig.update_layout(barmode="group", bargap=0.32, bargroupgap=0.14)
    fig.update_xaxes(range=[0, 1.08], title_text="Metric value", showgrid=True, gridcolor="#263347")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def confusion(data, normalize=True):
    counts = np.asarray(data["confusion_matrix"], dtype=int)
    ratios = counts / counts.sum(axis=1, keepdims=True)
    labels = ["Non-fraud", "Fraud"]
    fig = go.Figure(go.Heatmap(z=ratios if normalize else counts, x=labels, y=labels,
                              colorscale=[[0, "#1b2b40"], [1, MINT]], showscale=False, xgap=8, ygap=8,
                              customdata=counts, zmin=0, zmax=1 if normalize else int(counts.max()),
                              hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{customdata:,} transactions<extra></extra>"))
    for actual in range(2):
        for predicted in range(2):
            fraction = ratios[actual, predicted] if normalize else counts[actual, predicted] / counts.max()
            fig.add_annotation(x=labels[predicted], y=labels[actual], showarrow=False,
                               text=f"<b>{counts[actual,predicted]:,}</b><br>{ratios[actual,predicted]:.1%} of class",
                               font=dict(size=13, color="#082e27" if fraction > .55 else "#e6effc"))
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
        number=dict(valueformat=".4f", font=dict(size=40, color="#edf5ff")),
        gauge=dict(axis=dict(range=[0, 1], tickvals=[0, .5, 1], tickcolor="#8398b3", tickfont=dict(size=11)),
                   bar=dict(color=AMBER if score >= threshold else MINT, thickness=.65),
                   bgcolor="#19283c", borderwidth=0,
                   steps=[dict(range=[0, threshold], color="#25483f"), dict(range=[threshold, 1], color="#504331")],
                   threshold=dict(line=dict(color="#f2f6fc", width=3), thickness=.9, value=threshold))))
    style(fig, height=225)
    fig.update_layout(margin=dict(l=24, r=24, t=28, b=12))
    return fig
