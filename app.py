"""Local Veyra dashboard: streamlit run app.py."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
st.set_page_config(page_title="Veyra · Fraud intelligence", page_icon="◈", layout="wide")

from veyra.dashboard.app import render_app

render_app(ROOT)
