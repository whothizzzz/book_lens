"""
Live web book fetching (Google Books / Open Library) and runtime classification.
"""

from __future__ import annotations

import numpy as np
import requests

from app.config import GOOGLE_API_KEY, get_logger

logger = get_logger("book_api", log_file="app.log")

GOOGLE_BOOKS_ENDPOINT = "https://www.googleapis.com/books/v1/volumes"
OPEN_LIBRARY_ENDPOINT = "https://openlibrary.org/search.json"
REQUEST_TIMEOUT_SECONDS = 5


# =============================================================================
# Fetching
# =============================================================================


def _extract_isbn13(identifiers: list[dict]) -> str | None:
    for identifier in identifiers:
        if identifier.get("type") == "ISBN_13":
            return identifier.get("identifier")
    return None


def _extract_year(published_date) -> int | None:
    if not published_date:
        return None
    try:
        return int(str(published_date)[:4])
    except ValueError:
        return None


def _fetch_google_books(query: str, max_results: int) -> list[dict]:
    """Fetch books from Google Books API."""
    params = {"q": query, "maxResults": max_results}
    if GOOGLE_API_KEY:
        params["key"] = GOOGLE_API_KEY
    try:
        response = requests.get(
            GOOGLE_BOOKS_ENDPOINT,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Google Books API request failed for query %r: %s", query, exc)
        return []

    books = []
    for item in response.json().get("items", []):
        info = item.get("volumeInfo", {})
        books.append({
            "isbn13": _extract_isbn13(info.get("industryIdentifiers", [])),
            "title": info.get("title", ""),
            "authors": "; ".join(info.get("authors", [])),
            "categories": "; ".join(info.get("categories", [])),
            "description": info.get("description", ""),
            "published_year": _extract_year(info.get("publishedDate")),
            "average_rating": info.get("averageRating"),
            "num_pages": info.get("pageCount"),
            "ratings_count": info.get("ratingsCount"),
            "thumbnail": info.get("imageLinks", {}).get("thumbnail"),
            "source": "google_books_live",
        })
    return books


def _fetch_open_library(query: str, max_results: int) -> list[dict]:
    """Fetch books from Open Library search API."""
    try:
        response = requests.get(
            OPEN_LIBRARY_ENDPOINT,
            params={"q": query, "limit": max_results},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Open Library API request failed for query %r: %s", query, exc)
        return []

    books = []
    for doc in response.json().get("docs", []):
        isbn_list = doc.get("isbn") or []
        cover_id = doc.get("cover_i")
        books.append({
            "isbn13": next((isbn for isbn in isbn_list if len(isbn) == 13), None),
            "title": doc.get("title", ""),
            "authors": "; ".join(doc.get("author_name", [])),
            "categories": "; ".join(doc.get("subject", [])[:5]),
            "description": "",
            "published_year": doc.get("first_publish_year"),
            "average_rating": None,
            "num_pages": None,
            "ratings_count": None,
            "thumbnail": f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else None,
            "source": "open_library_live",
        })
    return books


def fetch_live_books(query: str, max_results: int = 5) -> list[dict]:
    """Fetch deduplicated live book results from Google Books and Open Library."""
    fetched_books = _fetch_google_books(query, max_results) + _fetch_open_library(query, max_results)

    seen_keys = set()
    deduplicated_books = []
    for book in fetched_books:
        key = book["isbn13"] or book["title"].strip().lower()
        if key and key not in seen_keys:
            seen_keys.add(key)
            deduplicated_books.append(book)

    logger.info(
        "Fetched %d live books for %r (%d unique)",
        len(fetched_books), query, len(deduplicated_books),
    )
    return deduplicated_books


# =============================================================================
# Runtime Classification
# =============================================================================


def zero_shot_scores(zero_shot_pipeline, text: str, labels: list[str]) -> np.ndarray:
    """Compute per-label zero-shot NLI probabilities."""
    result = zero_shot_pipeline(text, candidate_labels=labels)
    score_by_label = dict(zip(result["labels"], result["scores"]))
    return np.array([score_by_label[label] for label in labels])


def _classify_one_task(
    text: str,
    embedding: np.ndarray,
    nn_model,
    zero_shot_pipeline,
    threshold: float,
    labels: list[str],
) -> tuple[str, str]:
    """Predict label using trained NeuralNetwork above threshold, or fallback to zero-shot."""
    if nn_model is not None:
        probabilities = nn_model.predict_proba(embedding.reshape(1, -1))[0]
        best_index = int(np.argmax(probabilities))
        if probabilities[best_index] >= threshold:
            return labels[best_index], "neural_network"

    scores = zero_shot_scores(zero_shot_pipeline, text, labels)
    return labels[int(np.argmax(scores))], "zero_shot"


def classify_live_book(
    book: dict,
    text: str,
    embedding: np.ndarray,
    category_model,
    emotion_model,
    zero_shot_pipeline,
    category_threshold: float,
    emotion_threshold: float,
    category_labels: list[str],
    emotion_labels: list[str],
) -> dict:
    """Classify category and emotion for a live-fetched book."""
    category, category_mechanism = _classify_one_task(
        text, embedding, category_model, zero_shot_pipeline, category_threshold, category_labels,
    )
    emotion, emotion_mechanism = _classify_one_task(
        text, embedding, emotion_model, zero_shot_pipeline, emotion_threshold, emotion_labels,
    )

    logger.info(
        "Classified %r - category=%s (%s), emotion=%s (%s)",
        book.get("title"), category, category_mechanism, emotion, emotion_mechanism,
    )

    return {
        **book,
        "category": category,
        "emotion": emotion,
        "category_mechanism": category_mechanism,
        "emotion_mechanism": emotion_mechanism,
    }
