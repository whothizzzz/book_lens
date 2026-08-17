"""
Plotly chart components for vector space, similarity, and model analytics.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from app.config import (
    ACCENT_COLOR,
    CORPUS_COLOR,
    LIVE_COLOR,
    LOCAL_COLOR,
    QUERY_COLOR,
    chart_text_color,
)


def _layout(dark_mode: bool, **overrides) -> dict:
    """Shared transparent Plotly layout matching active theme."""
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=12, color=chart_text_color(dark_mode)),
        margin=dict(l=20, r=20, t=48, b=20),
    )
    layout.update(overrides)
    return layout


def vector_space_3d(
    query_embedding: np.ndarray,
    selected_embedding: np.ndarray,
    other_result_embeddings: np.ndarray,
    other_result_titles: list[str],
    other_result_is_live: list[bool],
    corpus_embeddings: np.ndarray,
    dark_mode: bool = False,
) -> go.Figure:
    """3D PCA scatter plot showing query, results, and corpus background."""
    input_dim = query_embedding.shape[0]
    other_matrix = (
        other_result_embeddings.reshape(-1, input_dim)
        if len(other_result_embeddings) else np.empty((0, input_dim))
    )
    all_embeddings = np.vstack([
        query_embedding.reshape(1, -1),
        selected_embedding.reshape(1, -1),
        other_matrix,
        corpus_embeddings,
    ])

    pca = PCA(n_components=3, random_state=0)
    coordinates = pca.fit_transform(all_embeddings)
    explained_variance = pca.explained_variance_ratio_ * 100

    query_point = coordinates[0]
    selected_point = coordinates[1]
    other_points = coordinates[2:2 + len(other_matrix)]
    corpus_points = coordinates[2 + len(other_matrix):]

    figure = go.Figure()

    figure.add_trace(go.Scatter3d(
        x=corpus_points[:, 0], y=corpus_points[:, 1], z=corpus_points[:, 2], mode="markers",
        marker=dict(size=2.5, color=CORPUS_COLOR), name="Corpus sample",
        hovertemplate="Corpus book<extra></extra>",
    ))

    if len(other_points):
        marker_colors = [LIVE_COLOR if is_live else LOCAL_COLOR for is_live in other_result_is_live]
        figure.add_trace(go.Scatter3d(
            x=other_points[:, 0], y=other_points[:, 1], z=other_points[:, 2], mode="markers+text",
            text=[title[:28] for title in other_result_titles], textposition="top center", textfont=dict(size=8),
            marker=dict(size=8, color=marker_colors, opacity=0.85),
            name="Other results", hovertemplate="<b>%{text}</b><extra></extra>",
        ))
        for point in other_points:
            figure.add_trace(go.Scatter3d(
                x=[query_point[0], point[0]], y=[query_point[1], point[1]], z=[query_point[2], point[2]],
                mode="lines", line=dict(color="rgba(128,128,128,0.4)", width=1),
                showlegend=False, hoverinfo="skip",
            ))

    figure.add_trace(go.Scatter3d(
        x=[selected_point[0]], y=[selected_point[1]], z=[selected_point[2]], mode="markers+text",
        text=["Selected"], textposition="top center", textfont=dict(size=10, color=ACCENT_COLOR),
        marker=dict(size=13, color=ACCENT_COLOR, symbol="diamond"), name="Selected book",
    ))

    figure.add_trace(go.Scatter3d(
        x=[query_point[0]], y=[query_point[1]], z=[query_point[2]], mode="markers+text",
        text=["Query"], textposition="top center", textfont=dict(size=11, color=QUERY_COLOR),
        marker=dict(size=15, color=QUERY_COLOR, symbol="diamond"), name="Query",
    ))

    figure.update_layout(**_layout(
        dark_mode,
        title=f"Vector space (PCA, {explained_variance[0]:.0f}% + {explained_variance[1]:.0f}% + {explained_variance[2]:.0f}% variance)",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        height=520,
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
    ))
    return figure


def similarity_ranking(titles: list[str], similarities: list[float], selected_title: str, dark_mode: bool = False) -> go.Figure:
    """Horizontal bar chart showing semantic similarity to query."""
    order = np.argsort(similarities)
    sorted_titles = [titles[i] for i in order]
    sorted_similarities = [similarities[i] for i in order]
    colors = [ACCENT_COLOR if title == selected_title else LOCAL_COLOR for title in sorted_titles]

    figure = go.Figure(go.Bar(
        x=sorted_similarities, y=sorted_titles, orientation="h",
        marker=dict(color=colors),
        text=[f"{value:.3f}" for value in sorted_similarities], textposition="outside",
        hovertemplate="<b>%{y}</b> %{x:.4f}<extra></extra>",
    ))
    figure.update_layout(**_layout(dark_mode, title="Semantic similarity to query", xaxis=dict(range=[0, 1.08]), height=420))
    return figure


def classification_explainability_catalog(category_confidence: float, emotion_confidence: float, dark_mode: bool = False) -> go.Figure:
    """Confidence score bars for catalog book predictions."""
    figure = go.Figure(go.Bar(
        x=["category", "emotion"], y=[category_confidence, emotion_confidence],
        marker=dict(color=[LOCAL_COLOR, LOCAL_COLOR]),
        text=[f"{v:.2f}" for v in (category_confidence, emotion_confidence)], textposition="outside",
    ))
    figure.update_layout(**_layout(
        dark_mode, title="Classification confidence (offline teacher model)",
        yaxis=dict(range=[0, 1.1], title="confidence"), height=380,
    ))
    return figure


def classification_explainability_live(
    category_labels: list[str], category_scores: np.ndarray, category_mechanism: str,
    emotion_labels: list[str], emotion_scores: np.ndarray, emotion_mechanism: str,
    dark_mode: bool = False,
) -> go.Figure:
    """Full class probability bars for live web books."""
    figure = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Category ({category_mechanism})", f"Emotion ({emotion_mechanism})"),
    )
    figure.add_bar(x=category_labels, y=category_scores, marker=dict(color=LOCAL_COLOR), row=1, col=1)
    figure.add_bar(x=emotion_labels, y=emotion_scores, marker=dict(color=LIVE_COLOR), row=1, col=2)
    figure.update_layout(**_layout(dark_mode, showlegend=False, title="Classification confidence (live book)", height=380))
    return figure


def model_performance_bars(label_names: list[str], per_class: dict, metric: str, dark_mode: bool = False) -> go.Figure:
    """Per-class metric comparison bar chart."""
    values = [per_class.get(label, {}).get(metric, 0.0) for label in label_names]
    figure = go.Figure(go.Bar(
        x=label_names, y=values, marker=dict(color=LOCAL_COLOR),
        text=[f"{v:.2f}" for v in values], textposition="outside",
    ))
    figure.update_layout(**_layout(
        dark_mode, title=metric.replace("_", " ").title(), yaxis=dict(range=[0, 1.1]), height=340,
    ))
    return figure


def roc_curves(label_names: list[str], roc_data: dict, dark_mode: bool = False) -> go.Figure:
    """One-vs-rest ROC curves for classification tasks."""
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", line=dict(color=CORPUS_COLOR, dash="dot"),
        name="Chance", hoverinfo="skip",
    ))
    for label in label_names:
        curve = roc_data.get(label)
        if curve is None:
            continue
        figure.add_trace(go.Scatter(
            x=curve["fpr"], y=curve["tpr"], mode="lines",
            name=f"{label} (AUC={curve['auc']:.2f})",
        ))
    figure.update_layout(**_layout(
        dark_mode, title="ROC curves (one-vs-rest)",
        xaxis=dict(title="False positive rate", range=[0, 1]),
        yaxis=dict(title="True positive rate", range=[0, 1]),
        height=440, legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
    ))
    return figure


def emotion_profile(
    emotion_labels: list[str],
    book_profile: np.ndarray,
    catalog_average_profile: np.ndarray,
    dark_mode: bool = False,
) -> go.Figure:
    """Radar chart comparing a book's emotion distribution against catalog averages."""
    figure = go.Figure()
    figure.add_trace(go.Scatterpolar(
        r=book_profile, theta=emotion_labels, fill="toself", name="This book",
        line=dict(color=ACCENT_COLOR), fillcolor="rgba(249,115,22,0.2)",
    ))
    figure.add_trace(go.Scatterpolar(
        r=catalog_average_profile, theta=emotion_labels, fill="toself", name="Catalog average",
        line=dict(color=CORPUS_COLOR, dash="dot"), fillcolor="rgba(156,163,175,0.12)",
    ))
    figure.update_layout(**_layout(
        dark_mode, title="Emotion profile", height=420,
        polar={"radialaxis": {"visible": True}},
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
    ))
    return figure
