"""Shared visual components. Report-derived and user-entered text is escaped."""

from datetime import datetime
from html import escape
import math

import streamlit as st


def number(value, decimals: int = 0, suffix: str = "") -> str:
    try:
        value = float(value)
        return f"{value:,.{decimals}f}{suffix}" if math.isfinite(value) else "—"
    except (TypeError, ValueError):
        return "—"


def date_label(value) -> str:
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y · %H:%M UTC")
    except (TypeError, ValueError):
        return "Snapshot time unavailable"


def header(eyebrow: str, title: str, description: str, badge: str = "SAVED LOCAL REPORTS"):
    st.html(f'<div class="eyebrow">{escape(eyebrow)}</div><div class="page-heading"><div>'
            f'<h1>{escape(title)}</h1><p>{escape(description)}</p></div>'
            f'<div class="heading-status"><span class="pill">{escape(badge)}</span></div></div>')


def kpis(items: list[tuple[str, str, str]]):
    cards = "".join(f'<div class="kpi"><div class="kpi-label">{escape(label)}</div>'
                    f'<div class="kpi-value">{escape(value)}</div><div class="kpi-detail">{escape(detail)}</div></div>'
                    for label, value, detail in items)
    st.html(f'<div class="kpi-grid" style="--count:{len(items)}">{cards}</div>')


def panel(title: str, caption: str):
    st.html(f'<div class="panel-title">{escape(title)}</div><div class="panel-caption">{escape(caption)}</div>')


def insight(title: str, text: str, *, amber: bool = False):
    st.html(f'<div class="insight {"amber" if amber else ""}"><strong>{escape(title)}</strong><br>{escape(text)}</div>')


def empty(title: str, description: str):
    st.html(f'<div class="empty-state"><div class="empty-symbol">◇</div><strong>{escape(title)}</strong>{escape(description)}</div>')


def chart(figure, key: str):
    st.plotly_chart(figure, width="stretch", theme=None, key=key,
                    config={"displayModeBar": False, "scrollZoom": False, "responsive": True})


def footer(store):
    if store.issues:
        with st.expander("Local artifact availability"):
            for name, issue in store.issues.items():
                st.caption(f"{name}: {issue}")
    st.html('<div class="footer">VEYRA / FRAUD INTELLIGENCE &nbsp; · &nbsp; Local workspace<br>'
            'Analytics use saved aggregate reports. No raw datasets, training, or holdout evaluation run in this dashboard.</div>')
