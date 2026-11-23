"""Knowledge-base loader and schema validation (task A3).

Reads ``courses.json``, ``course_descriptions.md`` and ``registration_rules.md``
from ``knowledge_base/`` and checks them for consistency.

Public API (shared interface contract):
    load_courses(path=None) -> dict[str, Course]

Extra helpers:
    load_descriptions(), load_rules(), validate_course_records(), validate_kb(),
    parse_slot()
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"

REQUIRED_FIELDS = ("code", "title", "credits", "level", "prerequisites",
                   "offered", "timeslot", "tags")
VALID_SEMESTERS = ("Fall", "Spring")
VALID_LEVELS = (1, 2, 3, 4)
DAYS = ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")

_CODE_RE = re.compile(r"^[A-Z]{2,5}\d{3}$")
_SLOT_RE = re.compile(r"^(Sun|Mon|Tue|Wed|Thu|Fri|Sat) (\d{2}):(\d{2})-(\d{2}):(\d{2})$")
_COURSE_HEADING_RE = re.compile(r"^##\s+([A-Z]{2,5}\d{3})\s+-\s+(.+?)\s*$")
_RULE_HEADING_RE = re.compile(r"^###\s+(R\d+)\s+-\s+(.+?)\s*$")


class KBValidationError(ValueError):
    """Raised when the knowledge base is inconsistent. ``problems`` lists every issue."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("Knowledge base validation failed:\n - " + "\n - ".join(self.problems))


@dataclass
class Course:
    code: str
    title: str
    credits: int
    level: int
    prerequisites: list = field(default_factory=list)
    offered: list = field(default_factory=list)
    timeslot: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    description: str = ""

    def meetings(self):
        """Meeting times as (day, start_minute, end_minute) tuples."""
        return [parse_slot(s) for s in self.timeslot]


def parse_slot(slot: str):
    """Parse 'Sun 10:00-11:30' into ('Sun', 600, 690). Raises ValueError if malformed."""
    m = _SLOT_RE.match(slot) if isinstance(slot, str) else None
    if not m:
        raise ValueError(f"bad timeslot {slot!r}; expected 'Day HH:MM-HH:MM' (e.g. 'Sun 10:00-11:30')")
    day, h1, m1, h2, m2 = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
    if h1 > 23 or h2 > 23 or m1 > 59 or m2 > 59:
        raise ValueError(f"bad time in timeslot {slot!r}")
    start, end = h1 * 60 + m1, h2 * 60 + m2
    if start >= end:
        raise ValueError(f"timeslot {slot!r} must end after it starts")
    return day, start, end


def count_sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


# ---------------------------------------------------------------- validation
def validate_course_records(records) -> list:
    """Return a list of problem strings (empty list = the catalog is valid)."""
    problems = []
    if not isinstance(records, list) or not records:
        return ["courses.json must contain a non-empty JSON list of course objects"]

    codes, by_code = [], {}
    for i, rec in enumerate(records):
        where = f"record #{i + 1}"
        if not isinstance(rec, dict):
            problems.append(f"{where}: not a JSON object")
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            problems.append(f"{where} ({rec.get('code', '?')}): missing required field(s) {missing}")
        code = rec.get("code")
        if "code" in rec:
            if not isinstance(code, str) or not _CODE_RE.match(code):
                problems.append(f"{where}: invalid course code {code!r} (expected e.g. 'CS301')")
            else:
                where = code
                if code in by_code:
                    problems.append(f"duplicate course code {code}")
                by_code[code] = rec
                codes.append(code)
        if "title" in rec and not (isinstance(rec["title"], str) and rec["title"].strip()):
            problems.append(f"{where}: title must be a non-empty string")
        cr = rec.get("credits")
        if "credits" in rec and not (isinstance(cr, int) and not isinstance(cr, bool) and 1 <= cr <= 6):
            problems.append(f"{where}: credits must be an integer 1-6, got {cr!r}")
        lv = rec.get("level")
        if "level" in rec and not (isinstance(lv, int) and not isinstance(lv, bool) and lv in VALID_LEVELS):
            problems.append(f"{where}: level must be one of {VALID_LEVELS}, got {lv!r}")
        for key in ("prerequisites", "tags"):
            val = rec.get(key)
            if key in rec and not (isinstance(val, list) and all(isinstance(x, str) for x in val)):
                problems.append(f"{where}: {key} must be a list of strings")
        off = rec.get("offered")
        if "offered" in rec:
            if not (isinstance(off, list) and off and all(s in VALID_SEMESTERS for s in off)):
                problems.append(f"{where}: offered must be a non-empty list drawn from {VALID_SEMESTERS}, got {off!r}")
        ts = rec.get("timeslot")
        if "timeslot" in rec:
            if not (isinstance(ts, list) and ts):
                problems.append(f"{where}: timeslot must be a non-empty list")
            else:
                for s in ts:
                    try:
                        parse_slot(s)
                    except ValueError as exc:
                        problems.append(f"{where}: {exc}")

    # cross-record checks
    for code, rec in by_code.items():
        prereqs = rec.get("prerequisites")
        if not isinstance(prereqs, list):
            continue
        for p in prereqs:
            if p == code:
                problems.append(f"{code}: lists itself as a prerequisite")
            elif p not in by_code:
                problems.append(f"{code}: prerequisite {p} does not exist in the catalog")
            else:
                plevel, clevel = by_code[p].get("level"), rec.get("level")
                if isinstance(plevel, int) and isinstance(clevel, int) and plevel > clevel:
                    problems.append(f"{code}: prerequisite {p} (level {plevel}) is a higher level than the course (level {clevel})")

    # cycle detection (depth-first search)
    state = {}
    def visit(c, path):
        if state.get(c) == 2:
            return
        if state.get(c) == 1:
            problems.append("prerequisite cycle: " + " -> ".join(path + [c]))
            return
        state[c] = 1
        for p in by_code.get(c, {}).get("prerequisites", []) or []:
            if p in by_code:
                visit(p, path + [c])
        state[c] = 2
    for c in list(by_code):
        visit(c, [])
    return problems


# ------------------------------------------------------------------- loaders
def load_descriptions(path=None) -> dict:
    """Parse course_descriptions.md into {code: description text}."""
    path = Path(path) if path else KB_DIR / "course_descriptions.md"
    out, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _COURSE_HEADING_RE.match(line)
        if m:
            current = m.group(1)
            out[current] = []
        elif current and line.strip():
            out[current].append(line.strip())
    return {c: " ".join(lines) for c, lines in out.items()}


def load_rules(path=None) -> dict:
    """Parse registration_rules.md into {'R1': {'title': ..., 'text': ...}, ...}."""
    path = Path(path) if path else KB_DIR / "registration_rules.md"
    rules, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _RULE_HEADING_RE.match(line)
        if m:
            current = m.group(1)
            rules[current] = {"title": m.group(2), "lines": []}
        elif line.startswith("#"):
            current = None
        elif current and line.strip():
            rules[current]["lines"].append(line.strip())
    return {r: {"title": v["title"], "text": " ".join(v["lines"])} for r, v in rules.items()}


def load_courses(path=None, with_descriptions: bool = True) -> dict:
    """Load and validate the course catalog. Returns {code: Course}.

    Raises KBValidationError listing every problem if the catalog is invalid.
    """
    path = Path(path) if path else KB_DIR / "courses.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    problems = validate_course_records(records)
    if problems:
        raise KBValidationError(problems)
    descriptions = {}
    desc_path = path.parent / "course_descriptions.md"
    if with_descriptions and desc_path.exists():
        descriptions = load_descriptions(desc_path)
    courses = {}
    for rec in records:
        courses[rec["code"]] = Course(
            code=rec["code"], title=rec["title"], credits=rec["credits"], level=rec["level"],
            prerequisites=list(rec["prerequisites"]), offered=list(rec["offered"]),
            timeslot=list(rec["timeslot"]), tags=list(rec["tags"]),
            description=descriptions.get(rec["code"], ""),
        )
    return courses


def validate_kb(kb_dir=None) -> dict:
    """Validate all three KB files together. Raises KBValidationError on any problem.

    Checks: catalog schema, descriptions (one per course, 4-6 sentences, no unknown codes),
    rules (R1..Rn numbered without gaps).  Returns a small summary dict.
    """
    kb_dir = Path(kb_dir) if kb_dir else KB_DIR
    problems = []
    records = json.loads((kb_dir / "courses.json").read_text(encoding="utf-8"))
    problems += validate_course_records(records)
    codes = {r.get("code") for r in records if isinstance(r, dict)}

    descs = load_descriptions(kb_dir / "course_descriptions.md")
    for c in sorted(codes - set(descs)):
        problems.append(f"{c}: no description in course_descriptions.md")
    for c in sorted(set(descs) - codes):
        problems.append(f"course_descriptions.md describes unknown course {c}")
    for c, text in descs.items():
        n = count_sentences(text)
        if not 4 <= n <= 6:
            problems.append(f"{c}: description has {n} sentences (expected 4-6)")

    rules = load_rules(kb_dir / "registration_rules.md")
    if not rules:
        problems.append("registration_rules.md contains no rules")
    expected_ids = [f"R{i}" for i in range(1, len(rules) + 1)]
    if list(rules) != expected_ids:
        problems.append(f"rule IDs must be R1..R{len(rules)} in order, found {list(rules)}")

    if problems:
        raise KBValidationError(problems)
    return {"courses": len(records), "descriptions": len(descs), "rules": len(rules)}
