"""Dashboard shell with native Streamlit page navigation."""

from pathlib import Path

import streamlit as st

from .artifacts import Artifacts
from .ui import footer
from .views import overview, performance, explainability


def render_app(root: Path):
    assets = Path(__file__).parent
    st.html(f"<style>{(assets / 'theme.css').read_text(encoding='utf-8')}</style>")
    st.logo(str(assets / "assets" / "brand.svg"), size="large", icon_image=str(assets / "assets" / "mark.svg"))
    store = Artifacts(root)

    def overview_page():
        overview(store)

    def performance_page():
        performance(store)

    def explainability_page():
        explainability(store)

    def predictor_page():
        from .predictor import predictor
        predictor(store)

    navigation = st.navigation({
        "INTELLIGENCE": [
            st.Page(overview_page, title="Overview", icon=":material/dashboard:", default=True),
            st.Page(performance_page, title="Model Performance", icon=":material/monitoring:", url_path="performance"),
            st.Page(explainability_page, title="Explainability", icon=":material/insights:", url_path="explainability"),
        ],
        "WORKSPACE": [st.Page(predictor_page, title="Risk Predictor", icon=":material/travel_explore:", url_path="predictor")],
    }, expanded=True)
    with st.sidebar:
        st.html('<div class="sidebar-card"><span class="pill">LOCAL WORKSPACE</span><br><br>'
                '<b>Evidence, not estimates.</b><br>Saved reports. Frozen model.<br>No live feed or background training.</div>')
        if st.button("Refresh saved reports", icon=":material/refresh:", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.html('<div class="sidebar-label">PRIVATE BY DEFAULT</div>')
        st.caption("Local files only. Manual inputs are not saved to disk.")
    navigation.run()
    footer(store)
