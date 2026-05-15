from __future__ import annotations

from dataclasses import dataclass


DOCUMENT_KEYWORDS = (
    "திருக்குறள்",
    "குறள்",
    "வள்ளுவர்",
    "இலக்கியம்",
    "பாடல்",
    "நூல்",
    "ஆவணம்",
    "document",
    "pdf",
    "file",
    "source",
)


@dataclass(frozen=True)
class QueryRouteResult:
    """Represents the selected route for an incoming user query."""

    route: str


class QueryRouter:
    """Lightweight keyword router to decide between RAG and direct LLM chat."""

    def route(self, question: str) -> QueryRouteResult:
        lowered = question.lower()
        for keyword in DOCUMENT_KEYWORDS:
            if keyword.lower() in lowered:
                return QueryRouteResult(route="document_related")
        return QueryRouteResult(route="general_chat")
