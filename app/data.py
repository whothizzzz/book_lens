"""
Data and model caching utilities for the Streamlit dashboard.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from app.config import (
    CATALOG_CSV,
    CATEGORY_MODEL_PATH,
    EMBEDDING_MODEL_NAME,
    EMOTION_MODEL_PATH,
    EMOTION_PROFILE_PATH,
    METRICS_PATH,
    ONLINE_ZERO_SHOT_MODEL_NAME,
    VECTOR_STORE_COLLECTION,
    VECTOR_STORE_DIR,
    get_logger,
)
from src.ml_from_scratch import NeuralNetwork

logger = get_logger("data", log_file="app.log")

DEFAULT_CONFIDENCE_THRESHOLD = 0.6


@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    """Load the full catalog dataset."""
    catalog = pd.read_csv(CATALOG_CSV, dtype={"isbn13": "string"})
    logger.info("Loaded catalog: %d rows", len(catalog))
    return catalog


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model() -> SentenceTransformer:
    """Load the shared SentenceTransformer model for query and catalog embedding."""
    logger.info("Loading embedding model %s", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_vector_store():
    """Load the persisted Chroma collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection = client.get_collection(VECTOR_STORE_COLLECTION)
    logger.info("Opened vector store %r (%d vectors)", VECTOR_STORE_COLLECTION, collection.count())
    return collection


def get_catalog_embedding(row_id: str, vector_store) -> np.ndarray:
    """Retrieve stored embedding for a specific catalog row ID."""
    result = vector_store.get(ids=[str(row_id)], include=["embeddings"])
    return np.asarray(result["embeddings"][0])


@st.cache_resource(show_spinner=False)
def load_category_model() -> NeuralNetwork | None:
    """Load trained category classifier, or None if unpopulated."""
    if not CATEGORY_MODEL_PATH.exists():
        logger.warning("Missing %s - category classification will use zero-shot fallback", CATEGORY_MODEL_PATH)
        return None
    return NeuralNetwork.load(CATEGORY_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_emotion_model() -> NeuralNetwork | None:
    """Load trained emotion classifier, or None if unpopulated."""
    if not EMOTION_MODEL_PATH.exists():
        logger.warning("Missing %s - emotion classification will use zero-shot fallback", EMOTION_MODEL_PATH)
        return None
    return NeuralNetwork.load(EMOTION_MODEL_PATH)


@st.cache_resource(show_spinner="Loading zero-shot classifier...")
def load_zero_shot_classifier():
    """Load lightweight zero-shot classification pipeline for runtime fallbacks."""
    from transformers import pipeline

    logger.info("Loading zero-shot classifier %s", ONLINE_ZERO_SHOT_MODEL_NAME)
    return pipeline("zero-shot-classification", model=ONLINE_ZERO_SHOT_MODEL_NAME)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    """Load evaluation metrics and confidence thresholds from notebook 05."""
    if not METRICS_PATH.exists():
        logger.warning("Missing %s - using default thresholds", METRICS_PATH)
        return {
            "threshold_category": DEFAULT_CONFIDENCE_THRESHOLD,
            "threshold_emotion": DEFAULT_CONFIDENCE_THRESHOLD,
        }
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_catalog_average_emotion_profile() -> np.ndarray | None:
    """Load catalog-wide average emotion score vector, if present."""
    if not EMOTION_PROFILE_PATH.exists():
        return None
    return np.load(EMOTION_PROFILE_PATH)
