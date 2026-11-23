"""Parsing and chunking of the knowledge base (task B1).

Turns the three knowledge-base files into small, self-contained text chunks that
can be embedded and retrieved.  Every chunk carries metadata (source file,
course code or rule ID) so each recommendation can cite exactly where its
evidence came from.

    course chunk  : one per course  - catalog facts from courses.json + the description
    rule chunk    : one per rule    - R1 ... R9 from registration_rules.md
    terms chunk   : the "Terms" section of registration_rules.md

Long texts (none in the current catalog) are split into overlapping word
windows by ``chunk_text`` so the pipeline still works if the documents grow.

Public API:
    build_chunks(kb_dir=None) -> list[Chunk]
    chunk_text(text, max_words=200, overlap=40) -> list[str]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from kb_loader import KB_DIR, load_courses, load_rules

MAX_WORDS = 200      # about 260-300 tokens for English text
OVERLAP = 40


@dataclass
class Chunk:
    id: str                 # unique, e.g. "course:CS301#0" or "rule:R5#0"
    text: str               # the text that is embedded and shown as evidence
    source: str             # citation, e.g. "course_descriptions.md#CS301"
    kind: str               # "course" | "rule" | "terms"
    ref: str                # course code, rule ID or "terms"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def chunk_text(text: str, max_words: int = MAX_WORDS, overlap: int = OVERLAP) -> list:
    """Split text into windows of at most ``max_words`` words that overlap by ``overlap`` words."""
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()] if text.strip() else []
    step = max(1, max_words - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
    return out


def _course_text(c) -> str:
    prereq = ", ".join(c.prerequisites) if c.prerequisites else "none"
    return (
        f"{c.code} - {c.title}. Level {c.level} course, {c.credits} credit hours. "
        f"Offered in: {', '.join(c.offered)}. Prerequisites: {prereq}. "
        f"Meets: {'; '.join(c.timeslot)}. Topics: {', '.join(c.tags)}.\n"
        f"{c.description}"
    ).strip()


def _terms_text(kb_dir: Path) -> str:
    """The '## Terms' section of registration_rules.md (definitions used by all rules)."""
    lines, inside = [], False
    for line in (kb_dir / "registration_rules.md").read_text(encoding="utf-8").splitlines():
        if re.match(r"^##\s+Terms", line):
            inside = True
            continue
        if inside and line.startswith("#"):
            break
        if inside and line.strip():
            lines.append(line.strip().lstrip("- ").replace("**", ""))
    return " ".join(lines)


def build_chunks(kb_dir=None) -> list:
    """Parse the knowledge base and return all chunks with their metadata."""
    kb_dir = Path(kb_dir) if kb_dir else KB_DIR
    courses = load_courses(kb_dir / "courses.json")
    rules = load_rules(kb_dir / "registration_rules.md")
    chunks = []

    for c in courses.values():
        meta = {"code": c.code, "title": c.title, "level": c.level, "credits": c.credits,
                "offered": c.offered, "tags": c.tags, "files": ["courses.json", "course_descriptions.md"]}
        for i, piece in enumerate(chunk_text(_course_text(c))):
            chunks.append(Chunk(id=f"course:{c.code}#{i}", text=piece,
                                source=f"course_descriptions.md#{c.code}", kind="course",
                                ref=c.code, metadata=meta))

    for rid, r in rules.items():
        text = f"Registration rule {rid} - {r['title']}. {r['text']}"
        for i, piece in enumerate(chunk_text(text)):
            chunks.append(Chunk(id=f"rule:{rid}#{i}", text=piece,
                                source=f"registration_rules.md#{rid}", kind="rule",
                                ref=rid, metadata={"rule_id": rid, "title": r["title"],
                                                   "files": ["registration_rules.md"]}))

    terms = _terms_text(kb_dir)
    for i, piece in enumerate(chunk_text(terms)):
        chunks.append(Chunk(id=f"terms#{i}", text="Registration terms and definitions. " + piece,
                            source="registration_rules.md#terms", kind="terms", ref="terms",
                            metadata={"files": ["registration_rules.md"]}))
    return chunks


if __name__ == "__main__":
    cs = build_chunks()
    print(f"{len(cs)} chunks")
    for c in cs[:2] + cs[-2:]:
        print("-" * 60)
        print(c.id, "|", c.source)
        print(c.text)
