"""Student profile: data class, validation and interactive input (task A4).

Public API (shared interface contract):
    load_profile(path) -> Profile

Validation never guesses.  A missing or invalid value becomes ``None`` and a clear
flag (rule R9) is added to ``Profile.flags``; the rule checker then refuses to
approve anything that depends on that value.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from kb_loader import load_courses, VALID_SEMESTERS, VALID_LEVELS


@dataclass
class Profile:
    id: str | None = None
    name: str | None = None
    level: int | None = None
    gpa: float | None = None
    passed: list | None = None          # None = missing, [] = a new student with no passed courses
    skills: list = field(default_factory=list)
    interests: list = field(default_factory=list)
    desired_credits: int | None = None
    semester: str | None = None
    flags: list = field(default_factory=list)   # filled by validate(); not part of the input file

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("flags")
        return d

    @classmethod
    def from_dict(cls, data: dict, courses: dict | None = None) -> "Profile":
        """Build a Profile from a dict (e.g. parsed JSON) and validate it."""
        p = cls(
            id=_clean_str(data.get("id")),
            name=_clean_str(data.get("name")),
            level=data.get("level"),
            gpa=data.get("gpa"),
            passed=data.get("passed"),
            skills=data.get("skills") or [],
            interests=data.get("interests") or [],
            desired_credits=data.get("desired_credits"),
            semester=data.get("semester"),
        )
        p.validate(courses)
        return p

    def validate(self, courses: dict | None = None) -> list:
        """Check the profile, normalise it, and (re)build ``self.flags``. Returns the flags."""
        courses = courses if courses is not None else load_courses()
        flags = []

        # level -------------------------------------------------------
        if _is_blank(self.level):
            self.level = None
            flags.append("R9: missing field 'level' - level eligibility (R4) cannot be verified")
        elif not (_is_int(self.level) and self.level in VALID_LEVELS):
            flags.append(f"R9: invalid level {self.level!r} (must be one of {VALID_LEVELS}) - level eligibility (R4) cannot be verified")
            self.level = None

        # gpa ---------------------------------------------------------
        if _is_blank(self.gpa):
            self.gpa = None
            flags.append("R9: missing field 'gpa' - overload (R2) and probation (R3) status cannot be determined; only the base limits of R1 are applied")
        elif not (_is_number(self.gpa) and 0.0 <= self.gpa <= 4.0):
            flags.append(f"R9: invalid gpa {self.gpa!r} (must be between 0.0 and 4.0) - overload (R2) and probation (R3) status cannot be determined; only the base limits of R1 are applied")
            self.gpa = None
        else:
            self.gpa = float(self.gpa)

        # semester ----------------------------------------------------
        if _is_blank(self.semester):
            self.semester = None
            flags.append("R9: missing field 'semester' - availability (R6) cannot be verified")
        else:
            sem = str(self.semester).strip().capitalize()
            if sem in VALID_SEMESTERS:
                self.semester = sem
            else:
                flags.append(f"R9: invalid semester {self.semester!r} (must be one of {VALID_SEMESTERS}) - availability (R6) cannot be verified")
                self.semester = None

        # passed courses ---------------------------------------------
        if self.passed is None:
            flags.append("R9: missing field 'passed' - prerequisites (R5) and repeated courses (R8) cannot be verified")
        elif not (isinstance(self.passed, list) and all(isinstance(c, str) for c in self.passed)):
            flags.append("R9: 'passed' must be a list of course codes - prerequisites (R5) and repeated courses (R8) cannot be verified")
            self.passed = None
        else:
            normalised = []
            for c in self.passed:
                c = c.strip().upper()
                if c and c not in normalised:
                    normalised.append(c)
            self.passed = normalised
            unknown = [c for c in self.passed if c not in courses]
            if unknown:
                flags.append(f"R9: unknown course code(s) in 'passed': {', '.join(unknown)} (not in the catalog)")

        # desired credits --------------------------------------------
        if _is_blank(self.desired_credits):
            self.desired_credits = None
            flags.append("R9: missing field 'desired_credits' - the preferred credit load is unknown")
        elif not (_is_int(self.desired_credits) and self.desired_credits > 0):
            flags.append(f"R9: invalid desired_credits {self.desired_credits!r} (must be a positive whole number)")
            self.desired_credits = None

        # skills / interests -----------------------------------------
        for key in ("skills", "interests"):
            val = getattr(self, key)
            if not (isinstance(val, list) and all(isinstance(x, str) for x in val)):
                flags.append(f"R9: '{key}' must be a list of strings; the value was ignored")
                setattr(self, key, [])

        self.flags = flags
        return flags


def load_profile(path, courses: dict | None = None) -> Profile:
    """Load a profile from a JSON file and validate it."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Profile.from_dict(data, courses)


def load_profiles(directory, courses: dict | None = None) -> list:
    """Load every student_*.json file in a directory (sorted by file name)."""
    return [load_profile(p, courses) for p in sorted(Path(directory).glob("student_*.json"))]


def prompt_profile(input_fn=input, print_fn=print, courses: dict | None = None) -> Profile:
    """Interactive form for a new student.  Press Enter to skip a question; the
    skipped field is then flagged as missing instead of being guessed.

    ``input_fn`` / ``print_fn`` can be replaced for testing or for a notebook widget.
    """
    courses = courses if courses is not None else load_courses()
    print_fn("Student profile (press Enter to skip any question)")

    def ask(label):
        return input_fn(f"{label}: ").strip()

    def to_number(text, kind):
        if not text:
            return None
        try:
            return kind(text)
        except ValueError:
            return text      # kept as-is so validate() reports it as invalid

    def to_list(text):
        return [x.strip() for x in text.split(",") if x.strip()]

    data = {}
    data["id"] = ask("Student ID") or None
    data["name"] = ask("Name") or None
    data["level"] = to_number(ask("Level (1-4)"), int)
    data["gpa"] = to_number(ask("Cumulative GPA (0.0-4.0)"), float)
    raw_passed = ask("Passed course codes, comma-separated (type 'none' if you have passed none)")
    if raw_passed.lower() == "none":
        data["passed"] = []
    elif raw_passed:
        data["passed"] = to_list(raw_passed)
    else:
        data["passed"] = None
    data["skills"] = to_list(ask("Skills, comma-separated"))
    data["interests"] = to_list(ask("Interests / goals, comma-separated"))
    data["desired_credits"] = to_number(ask("Desired credit hours"), int)
    data["semester"] = ask("Semester to register for (Fall/Spring)") or None
    profile = Profile.from_dict(data, courses)
    if profile.flags:
        print_fn("Please note:")
        for f in profile.flags:
            print_fn("  - " + f)
    return profile


# ----------------------------------------------------------------- helpers
def _clean_str(x):
    return str(x).strip() if x not in (None, "") else None


def _is_blank(x) -> bool:
    return x is None or (isinstance(x, str) and not x.strip())


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)
