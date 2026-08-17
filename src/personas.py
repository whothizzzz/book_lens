"""
Reader persona presets and query biasing utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Persona:
    """Reading persona preset that steers retrieval and category preference."""

    name: str
    query_boost: str
    preferred_categories: tuple[str, ...]
    tone: str
    pinned_works: tuple[str, ...] = field(default_factory=tuple)
    icon: str = ":material/person:"

    def boost_query(self, query: str) -> str:
        """Append persona bias terms to a user query."""
        query = query.strip()
        if not self.query_boost:
            return query
        return f"{query} {self.query_boost}".strip()

    def match_score(self, category: str) -> int:
        """Calculate match score (0-100) between a category and this persona."""
        if not self.preferred_categories:
            return 60
        return 100 if category in self.preferred_categories else 30


PERSONAS: dict[str, Persona] = {
    "General Reader": Persona(
        name="General Reader",
        query_boost="",
        preferred_categories=(),
        tone="balanced, broad perspective",
        icon=":material/menu_book:",
    ),
    "Engineering Student": Persona(
        name="Engineering Student",
        query_boost="technical, academic, STEM-focused",
        preferred_categories=("Academia",),
        tone="technical and analytical",
        icon=":material/engineering:",
    ),
    "Nepali Literature Enthusiast": Persona(
        name="Nepali Literature Enthusiast",
        query_boost="Nepali poetry, epic drama, philosophy, romanticism",
        preferred_categories=("Nepali Literature",),
        tone="reflective and lyrical",
        pinned_works=(
            "Muna Madan - Laxmi Prasad Devkota",
            "Shirishko Phool - Parijat",
            "Palpasa Cafe - Narayan Wagle",
        ),
        icon=":material/auto_stories:",
    ),
    "Classic Drama Enthusiast": Persona(
        name="Classic Drama Enthusiast",
        query_boost="classic tragedy, drama, Renaissance and Elizabethan literature",
        preferred_categories=("Fiction",),
        tone="dramatic and classical",
        pinned_works=(
            "Hamlet - William Shakespeare",
            "Doctor Faustus - Christopher Marlowe",
            "Oedipus Rex - Sophocles",
        ),
        icon=":material/theater_comedy:",
    ),
}


def get_persona(name: str) -> Persona:
    """Retrieve persona by name, defaulting to General Reader."""
    return PERSONAS.get(name, PERSONAS["General Reader"])


def persona_names() -> list[str]:
    """List all available persona names."""
    return list(PERSONAS.keys())
