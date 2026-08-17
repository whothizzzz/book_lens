"""
Streamlit page views for BookLens: Search, Analytics, and Model Performance.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import streamlit as st

from app import charts, components
from app import data as app_data
from app.config import (
    CATEGORIES,
    EMBEDDING_PASSAGE_PREFIX,
    EMBEDDING_QUERY_PREFIX,
    EMOTIONS,
    build_passage_text,
    get_logger,
    is_present,
)
from app.ml_models import ClassifierBundle
from src.book_api import fetch_live_books
from src.personas import Persona

logger = get_logger("pages", log_file="app.log")

LOCAL_RESULT_COUNT = 10
CORPUS_SAMPLE_SIZE = 300

_WORD_PATTERN = re.compile(r"[a-zA-Z']+")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "with",
    "is", "are", "was", "were", "by", "at", "as", "it", "its", "this",
    "that", "about", "from",
}


def keyword_overlap(query: str, book_text: str) -> list[str]:
    """Extract non-stopword tokens shared between query and book text."""
    query_words = set(_WORD_PATTERN.findall(query.lower())) if query else set()
    book_words = set(_WORD_PATTERN.findall(book_text.lower())) if book_text else set()
    return sorted((query_words & book_words) - _STOPWORDS)


def _keyword_overlap_count(query: str, book: dict) -> int:
    title = book.get("title")
    description = book.get("description")
    book_text = f"{title if is_present(title) else ''} {description if is_present(description) else ''}"
    return len(keyword_overlap(query, book_text))


def search_catalog(
    query: str,
    persona: Persona,
    embedding_model,
    vector_store,
    catalog: pd.DataFrame,
) -> tuple[list[dict], np.ndarray]:
    """Search catalog using vector similarity re-ranked by persona preferences and keyword overlap."""
    boosted_query = persona.boost_query(query)
    query_embedding = embedding_model.encode(
        [EMBEDDING_QUERY_PREFIX + boosted_query], normalize_embeddings=True,
    )[0]

    results = vector_store.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=LOCAL_RESULT_COUNT,
        include=["embeddings", "distances", "metadatas"],
    )

    books = []
    for row_id, distance, embedding in zip(
        results["ids"][0], results["distances"][0], results["embeddings"][0],
    ):
        book = catalog.iloc[int(row_id)].to_dict()
        book["_catalog_row_id"] = row_id
        book["similarity"] = 1.0 - distance
        book["_embedding"] = np.asarray(embedding)
        books.append(book)

    def sort_key(book: dict):
        keyword_score = _keyword_overlap_count(query, book)
        persona_match = book.get("category") in persona.preferred_categories
        return (-keyword_score, -int(persona_match), -book["similarity"])

    books.sort(key=sort_key)

    top_similarity = books[0]["similarity"] if books else 0.0
    logger.info(
        "Search %r (persona=%s) -> %d local results, top similarity=%.3f",
        query, persona.name, len(books), top_similarity,
    )
    return books, query_embedding


def maybe_augment_with_live_results(
    query: str,
    query_embedding: np.ndarray,
    embedding_model,
    classifier_bundle: ClassifierBundle,
    live_search_enabled: bool = True,
) -> list[dict]:
    """Fetch and classify live web results if live search is enabled."""
    if not live_search_enabled:
        return []

    live_books = fetch_live_books(query)
    if not live_books:
        return []

    passage_texts = [
        EMBEDDING_PASSAGE_PREFIX + f"{book['title']} . {book.get('categories', '')} . {book.get('description', '')}"
        for book in live_books
    ]
    live_embeddings = embedding_model.encode(passage_texts, normalize_embeddings=True)

    augmented_books = []
    for book, embedding in zip(live_books, live_embeddings):
        classified_book = classifier_bundle.classify(book, embedding)
        classified_book["similarity"] = float(np.dot(query_embedding, embedding))
        classified_book["_embedding"] = embedding
        augmented_books.append(classified_book)

    logger.info("Augmented %r with %d live results", query, len(augmented_books))
    return augmented_books


def show_search_page(catalog, embedding_model, vector_store, classifier_bundle) -> None:
    """Render the primary search page."""
    title_col, nav_col = st.columns([5, 1])
    with title_col:
        st.title("BookLens")
        st.caption("Describe a book. Get a recommendation.")
    with nav_col:
        if st.button("Model Performance", icon=":material/monitoring:"):
            st.session_state.page = "model_performance"
            st.rerun()
    st.divider()

    persona = components.persona_selector()
    st.divider()

    query_text, submitted, live_search_enabled = components.search_bar()

    if submitted and query_text.strip():
        with st.spinner("Searching your catalog..."):
            local_results, query_embedding = search_catalog(
                query_text, persona, embedding_model, vector_store, catalog,
            )

        live_results = []
        if live_search_enabled:
            with st.spinner("Checking the web for more results..."):
                live_results = maybe_augment_with_live_results(
                    query_text, query_embedding, embedding_model, classifier_bundle,
                    live_search_enabled=live_search_enabled,
                )

        st.session_state.search_results = local_results + live_results
        st.session_state.search_query = query_text
        st.session_state.query_embedding = query_embedding

    if "search_results" not in st.session_state:
        return

    if not st.session_state.search_results:
        st.warning("No results found - try a different description.")
        return

    st.divider()
    selected_book = components.render_results_grid(st.session_state.search_results, persona)
    if selected_book is not None:
        st.session_state.selected_book = selected_book
        st.session_state.page = "analytics"
        st.rerun()


def show_analytics_page(catalog, vector_store, classifier_bundle, dark_mode: bool) -> None:
    """Render 4-tab book analytics view (Vector Space, Similarity, Explainability, Emotion)."""
    if "selected_book" not in st.session_state or "search_results" not in st.session_state:
        st.error("No book selected.")
        if st.button("Back to search", icon=":material/arrow_back:"):
            st.session_state.page = "search"
            st.rerun()
        return

    if st.button("Back to results", icon=":material/arrow_back:"):
        st.session_state.page = "search"
        st.rerun()

    query_embedding = st.session_state.query_embedding
    selected_book = st.session_state.selected_book
    all_results = st.session_state.search_results

    st.title(selected_book.get("title", "Selected book"))
    st.caption(selected_book.get("authors", ""))
    st.divider()

    tab_vector_space, tab_similarity, tab_explainability, tab_emotion = st.tabs([
        "3D Vector Space", "Semantic Similarity", "Classification Explainability", "Emotion Profile",
    ])

    selected_embedding = selected_book["_embedding"]
    selected_text = build_passage_text(
        selected_book.get("title"), selected_book.get("categories"), selected_book.get("description"),
    )
    is_live = str(selected_book.get("source", "")).endswith("_live")
    other_results = [book for book in all_results if book is not selected_book]

    with tab_vector_space:
        corpus_sample = catalog.sample(n=min(CORPUS_SAMPLE_SIZE, len(catalog)), random_state=0)
        corpus_embeddings = np.array([
            app_data.get_catalog_embedding(str(row_index), vector_store) for row_index in corpus_sample.index
        ])
        other_embeddings = (
            np.array([book["_embedding"] for book in other_results])
            if other_results else np.empty((0, selected_embedding.shape[0]))
        )
        figure = charts.vector_space_3d(
            query_embedding, selected_embedding, other_embeddings,
            [book.get("title", "") for book in other_results],
            [str(book.get("source", "")).endswith("_live") for book in other_results],
            corpus_embeddings, dark_mode=dark_mode,
        )
        st.plotly_chart(figure, width="stretch")

    with tab_similarity:
        figure = charts.similarity_ranking(
            [book.get("title", "") for book in all_results],
            [book["similarity"] for book in all_results],
            selected_book.get("title", ""), dark_mode=dark_mode,
        )
        st.plotly_chart(figure, width="stretch")

    with tab_explainability:
        if is_live:
            category_scores, category_mechanism = classifier_bundle.category_breakdown(selected_text, selected_embedding)
            emotion_scores, emotion_mechanism = classifier_bundle.emotion_breakdown(selected_text, selected_embedding)
            figure = charts.classification_explainability_live(
                CATEGORIES, category_scores, category_mechanism,
                EMOTIONS, emotion_scores, emotion_mechanism, dark_mode=dark_mode,
            )
        else:
            figure = charts.classification_explainability_catalog(
                selected_book.get("category_confidence", 0.0),
                selected_book.get("emotion_confidence", 0.0), dark_mode=dark_mode,
            )
        st.plotly_chart(figure, width="stretch")

    with tab_emotion:
        book_profile = classifier_bundle.emotion_profile(selected_book, selected_text)
        catalog_average_profile = app_data.load_catalog_average_emotion_profile()
        if catalog_average_profile is None:
            st.info("Catalog average emotion profile not available yet - run notebook 05.")
        else:
            figure = charts.emotion_profile(EMOTIONS, book_profile, catalog_average_profile, dark_mode=dark_mode)
            st.plotly_chart(figure, width="stretch")


def show_model_performance_page(dark_mode: bool) -> None:
    """Render offline validation metrics from notebook 05."""
    if st.button("Back to search", icon=":material/arrow_back:"):
        st.session_state.page = "search"
        st.rerun()

    st.title("Model Performance")
    st.caption("Held-out validation metrics from the most recent run of notebooks/05_train_classifiers.ipynb.")
    st.divider()

    metrics = app_data.load_metrics()
    if "category" not in metrics or "emotion" not in metrics:
        st.info("No training metrics found yet - run notebooks/05_train_classifiers.ipynb to produce models/metrics.json.")
        return

    for task_name, label_names in (("category", CATEGORIES), ("emotion", EMOTIONS)):
        task_metrics = metrics[task_name]
        st.subheader(task_name.title())
        st.caption(
            f"Training strategy: **{task_metrics.get('training_strategy', task_metrics.get('class_weight') or 'unweighted')}** "
            f"({task_metrics.get('n_training_rows', '?')} training rows after balancing) - "
            f"confidence threshold {task_metrics.get('threshold', 0):.2f} "
            f"(precision {task_metrics.get('threshold_precision', 0):.2f} at "
            f"{task_metrics.get('threshold_coverage_estimate', 0) * 100:.1f}% coverage)"
        )

        columns = st.columns(5)
        columns[0].metric("Accuracy", f"{task_metrics['accuracy']:.3f}")
        columns[1].metric("Precision (macro)", f"{task_metrics.get('precision_macro', 0):.3f}")
        columns[2].metric("Recall (macro)", f"{task_metrics.get('recall_macro', 0):.3f}")
        columns[3].metric("F1 (macro)", f"{task_metrics['macro_f1']:.3f}")
        columns[4].metric("ROC-AUC (macro)", f"{task_metrics.get('roc_auc_macro', 0):.3f}")
        st.caption(
            f"Majority-class baseline: accuracy {task_metrics['majority_baseline_accuracy']:.3f}, "
            f"macro-F1 {task_metrics['majority_baseline_macro_f1']:.3f}"
        )

        per_class = task_metrics.get("per_class", {})
        roc_curves_data = task_metrics.get("roc_curves", {})
        if per_class:
            bar_columns = st.columns(3)
            for column, metric_name in zip(bar_columns, ("precision", "recall", "f1")):
                with column:
                    st.plotly_chart(
                        charts.model_performance_bars(label_names, per_class, metric_name, dark_mode=dark_mode),
                        width="stretch",
                    )
        if roc_curves_data:
            st.plotly_chart(charts.roc_curves(label_names, roc_curves_data, dark_mode=dark_mode), width="stretch")

        st.divider()
