"""
Classification bundle combining neural network models with zero-shot fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app import data as app_data
from app.config import CATEGORIES, EMOTIONS, build_passage_text, is_present
from src.book_api import classify_live_book, zero_shot_scores
from src.ml_from_scratch import NeuralNetwork


@dataclass
class ClassifierBundle:
    """Encapsulates category and emotion classifiers with zero-shot fallbacks."""

    category_model: NeuralNetwork | None
    emotion_model: NeuralNetwork | None
    zero_shot_pipeline: object
    category_threshold: float
    emotion_threshold: float

    def classify(self, book: dict, embedding: np.ndarray) -> dict:
        """Classify a single live-fetched book."""
        text = build_passage_text(book.get("title"), book.get("categories"), book.get("description"))
        return classify_live_book(
            book, text, embedding,
            self.category_model, self.emotion_model, self.zero_shot_pipeline,
            self.category_threshold, self.emotion_threshold,
            CATEGORIES, EMOTIONS,
        )

    def category_breakdown(self, text: str, embedding: np.ndarray) -> tuple[np.ndarray, str]:
        """Return per-label category probabilities and active prediction mechanism."""
        if self.category_model is not None:
            return self.category_model.predict_proba(embedding.reshape(1, -1))[0], "neural_network"
        return zero_shot_scores(self.zero_shot_pipeline, text, CATEGORIES), "zero_shot"

    def emotion_breakdown(self, text: str, embedding: np.ndarray) -> tuple[np.ndarray, str]:
        """Return per-label emotion probabilities and active prediction mechanism."""
        if self.emotion_model is not None:
            return self.emotion_model.predict_proba(embedding.reshape(1, -1))[0], "neural_network"
        return zero_shot_scores(self.zero_shot_pipeline, text, EMOTIONS), "zero_shot"

    def emotion_profile(self, book: dict, text: str) -> np.ndarray:
        """Get or compute the full 4-class emotion profile for a book."""
        persisted = [book.get(f"emotion_score_{label}") for label in EMOTIONS]
        if all(is_present(value) for value in persisted):
            return np.array(persisted, dtype=float)
        return zero_shot_scores(self.zero_shot_pipeline, text, EMOTIONS)


def build_classifier_bundle() -> ClassifierBundle:
    """Instantiate the runtime classification bundle."""
    metrics = app_data.load_metrics()

    return ClassifierBundle(
        category_model=app_data.load_category_model(),
        emotion_model=app_data.load_emotion_model(),
        zero_shot_pipeline=app_data.load_zero_shot_classifier(),
        category_threshold=metrics.get("threshold_category", app_data.DEFAULT_CONFIDENCE_THRESHOLD),
        emotion_threshold=metrics.get("threshold_emotion", app_data.DEFAULT_CONFIDENCE_THRESHOLD),
    )
