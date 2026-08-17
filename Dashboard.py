"""
BookLens: Semantic Book Recommender
Streamlit application entry point and page router.
"""

from __future__ import annotations

import streamlit as st

from app import data as app_data
from app.config import dark_mode_active, get_logger
from app.ml_models import build_classifier_bundle
from app.pages import show_analytics_page, show_model_performance_page, show_search_page

logger = get_logger("dashboard", log_file="app.log")

st.set_page_config(page_title="BookLens", page_icon=":material/menu_book:", layout="wide")


def main() -> None:
    dark_mode = dark_mode_active()

    catalog = app_data.load_catalog()
    embedding_model = app_data.load_embedding_model()
    vector_store = app_data.load_vector_store()
    classifier_bundle = build_classifier_bundle()

    page = st.session_state.get("page", "search")
    if page == "analytics":
        show_analytics_page(catalog, vector_store, classifier_bundle, dark_mode)
    elif page == "model_performance":
        show_model_performance_page(dark_mode)
    else:
        show_search_page(catalog, embedding_model, vector_store, classifier_bundle)


if __name__ == "__main__":
    main()
