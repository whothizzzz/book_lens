"""
Streamlit UI components: persona selector, search bar, and result cards.
"""

from __future__ import annotations

import os
import re
import unicodedata

import streamlit as st

from app.config import get_logger, is_present
from src.personas import PERSONAS, Persona, persona_names

logger = get_logger("components", log_file="app.log")

FALLBACK_COVER = os.path.join("app", "no_cover_found.jpg")

_DEVANAGARI_PATTERN = re.compile(r"[ऀ-ॿ]")


def contains_devanagari(text: str) -> bool:
    """Check if text contains Devanagari characters."""
    return bool(text) and bool(_DEVANAGARI_PATTERN.search(text))


def romanize_nepali(text: str) -> str:
    """Phonetically transliterate Devanagari text into Roman-script Nepali."""
    if not contains_devanagari(text):
        return text

    from indic_transliteration import sanscript

    normalized_text = text.replace("॥", ".").replace("।", ".")
    iast_text = sanscript.transliterate(normalized_text, sanscript.DEVANAGARI, sanscript.IAST)

    decomposed = unicodedata.normalize("NFKD", iast_text)
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def persona_selector() -> Persona:
    """Render horizontal button bar for persona selection."""
    st.subheader("Reading persona")

    persona_id = st.session_state.get("selected_persona", "General Reader")
    columns = st.columns(len(PERSONAS))
    for column, name in zip(columns, persona_names()):
        with column:
            persona = PERSONAS[name]
            if st.button(
                persona.name, key=f"persona_{name}", icon=persona.icon,
                width="stretch", type="primary" if name == persona_id else "secondary",
            ):
                st.session_state.selected_persona = name
                st.rerun()

    active = PERSONAS[persona_id]
    st.caption(active.tone)
    if active.pinned_works:
        st.caption("You might already know: " + " | ".join(active.pinned_works))

    return active


_QUERY_INPUT_KEY = "query_input"


def search_bar() -> tuple[str, bool, bool]:
    """
    Search controls: a text input box and a toggle for whether live web
    results are allowed to supplement the catalog.

    Returns:
        (query_text, search_submitted, live_search_enabled).
    """
    if _QUERY_INPUT_KEY not in st.session_state:
        st.session_state[_QUERY_INPUT_KEY] = ""

    query_text = st.text_input(
        "Describe a book you're looking for",
        key=_QUERY_INPUT_KEY,
        placeholder="e.g. a young wizard discovers he has magical powers",
    )

    live_search_enabled = st.toggle(
        "Search the web too, when the catalog doesn't have a strong match",
        value=True,
        key="live_search_enabled",
    )

    submitted = st.button("Search", type="primary", icon=":material/search:")
    return query_text, submitted, live_search_enabled


def render_result_card(book: dict, rank: int, persona: Persona) -> bool:
    """
    One search result: cover (or a fallback image) beside title/authors/tags
    and a "View analytics" button, inside a bordered card.

    Returns:
        True if this card's "View analytics" button was just clicked.
    """
    with st.container(border=True):
        col_image, col_info = st.columns([1, 3], gap="small")

        with col_image:
            thumbnail = book.get("thumbnail")
            st.image(thumbnail if is_present(thumbnail) else FALLBACK_COVER, width=90)

        with col_info:
            title = book.get("title")
            st.markdown(f"**{title if is_present(title) else 'Untitled'}**")
            if is_present(title) and contains_devanagari(title):
                st.caption(f"romanized: {romanize_nepali(title)}")
            authors = book.get("authors")
            st.caption(authors if is_present(authors) else "Unknown author")

            is_live = str(book.get("source", "")).endswith("_live")
            category = book.get("category")
            emotion = book.get("emotion")

            tag_parts = []
            if is_present(category):
                tag_parts.append(f"`{category}`")
                tag_parts.append(f"Match: {persona.match_score(category)}% ({persona.name})")
            if is_present(emotion):
                tag_parts.append(f"Emotion: {emotion}")
            if "similarity" in book:
                tag_parts.append(f"Similarity: {book['similarity']:.2f}")
            if is_live:
                tag_parts.append(":material/language: Live web")
            st.caption(" | ".join(tag_parts))

            description = book.get("description")
            if is_present(description):
                snippet = description[:180] + ("..." if len(description) > 180 else "")
                st.caption(snippet)

            return st.button(
                "View analytics", key=f"select_{rank}_{book.get('title', '')}",
                icon=":material/analytics:",
            )


def render_results_grid(all_results: list[dict], persona: Persona) -> dict | None:
    """
    Render local and live results as a 2-column card grid, in separate
    sections so it's always clear which results came from the catalog
    versus the live web. Returns the book whose "View analytics" button
    was just clicked, or None.
    """
    local_results = [book for book in all_results if not str(book.get("source", "")).endswith("_live")]
    live_results = [book for book in all_results if str(book.get("source", "")).endswith("_live")]

    selected_book = None

    def render_section(books: list[dict], rank_offset: int) -> None:
        nonlocal selected_book
        for row_start in range(0, len(books), 2):
            columns = st.columns(2, gap="medium")
            for column_index, book in enumerate(books[row_start:row_start + 2]):
                rank = rank_offset + row_start + column_index + 1
                with columns[column_index]:
                    if render_result_card(book, rank, persona):
                        selected_book = book

    st.markdown(f"### In your catalog ({len(local_results)})")
    render_section(local_results, rank_offset=0)

    if live_results:
        st.divider()
        st.subheader("Also from the web")
        st.caption("Your catalog didn't have a strong match, so these were fetched live.")
        render_section(live_results, rank_offset=len(local_results))

    return selected_book
