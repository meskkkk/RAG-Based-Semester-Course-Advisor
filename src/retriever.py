"""Retriever (task B3).

Builds a search query from the student profile, finds the most relevant course
passages, and always attaches the registration-rule passages that decided each
course (looked up by rule ID, so the cited rule is never a "near miss").

Public API (shared interface contract):
    retrieve(query, k=5, kind=None) -> [{"text", "source", "score", "kind", "ref"}]

Extra helpers:
    profile_query(profile)            text query built from interests, skills and level
    rank_courses(profile)             every course code ranked by relevance to the student
    course_evidence(code)             the course's own chunk(s)
    rule_evidence(rule_ids)           the chunks of the given rules
"""
from __future__ import annotations

from vector_store import get_store

# A few synonyms so short interest words also match the catalog wording.
_EXPAND = {
    "ai": "artificial intelligence AI machine learning",
    "ml": "machine learning",
    "nlp": "natural language processing NLP text",
    "cv": "computer vision images",
    "computer vision": "computer vision image processing",
    "security": "security cybersecurity",
    "web": "web development websites",
    "mobile apps": "mobile application development android",
    "big data": "big data analytics data",
    "iot": "IoT embedded systems sensors",
}

LEVEL_BONUS = 0.03     # small preference for courses at the student's current level


def _hit(chunk, score) -> dict:
    return {"text": chunk.text, "source": chunk.source, "score": None if score is None else round(float(score), 4),
            "kind": chunk.kind, "ref": chunk.ref}


def retrieve(query: str, k: int = 5, kind: str | None = None) -> list:
    """Top-k passages for a free-text query (best first)."""
    store = get_store(verbose=False)
    return [_hit(c, s) for c, s in store.search(query, k=k, kind=kind)]


def profile_query(profile) -> str:
    """Turn a student profile into a search query."""
    parts = []
    for item in list(profile.interests or []) + list(profile.skills or []):
        parts.append(item)
        extra = _EXPAND.get(item.strip().lower())
        if extra:
            parts.append(extra)
    if profile.level:
        parts.append(f"level {profile.level} course")
    if profile.semester:
        parts.append(f"offered in {profile.semester}")
    return ". ".join(parts) if parts else "course"


def rank_courses(profile) -> list:
    """All course codes ranked by relevance to the student: [(code, score)], best first.

    score = cosine similarity of the profile query and the course passage,
            plus a small bonus for courses at the student's own level.
    Courses the student already passed are left out.
    """
    store = get_store(verbose=False)
    scores = store.scores_for(profile_query(profile), kind="course")
    passed = set(profile.passed or [])
    ranked = []
    for code, s in scores.items():
        if code in passed:
            continue
        level = store.get(code)[0].metadata.get("level")
        bonus = LEVEL_BONUS if profile.level and level == profile.level else 0.0
        ranked.append((code, round(s + bonus, 4)))
    ranked.sort(key=lambda t: -t[1])
    return ranked


def course_evidence(code: str, score=None) -> list:
    store = get_store(verbose=False)
    return [_hit(c, score) for c in store.get(code)]


def rule_evidence(rule_ids) -> list:
    """Rule passages by ID, in the order given, without duplicates."""
    store = get_store(verbose=False)
    out, seen = [], set()
    for rid in rule_ids:
        if rid in seen:
            continue
        seen.add(rid)
        out += [_hit(c, None) for c in store.get(rid)]
    return out
