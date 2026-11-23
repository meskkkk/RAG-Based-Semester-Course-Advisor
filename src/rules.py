"""Rule checker: explicit Python checks for the registration rules (task A6).

Every rejection carries a human-readable reason and the ID of the rule in
``knowledge_base/registration_rules.md`` that caused it.  If a rule or a
required field is missing, the result carries a flag (rule R9) and the course
is NOT approved - nothing is ever guessed.

Public API (shared interface contract):
    check_course(profile, code)       -> {"eligible", "reasons", "rule_refs", "missing"}
    check_plan(profile, [codes])      -> {"accepted", "rejected", "total_credits", "flags"}

Additive extras (existing keys unchanged, so Member B's code keeps working):
    check_course(...)["flags"]            unverifiable items (a subset of "reasons")
    check_plan(...)["credit_limits"]      {"min", "max", "rule", "status"} that applied
    eligible_courses(), credit_limits(), find_clashes(), catalog_clashes(),
    verify_rule_references(), format_plan_result()

Result shapes
    accepted : list of course codes (strings), in the order the student listed them
    rejected : list of {"code", "reasons", "rule_refs", "missing"}  (missing = unmet prerequisite codes)
"""
from __future__ import annotations

from functools import lru_cache

from kb_loader import Course, load_courses, load_rules

# ----- numbers from registration_rules.md (a test checks they appear in the document) ----
BASE_MIN_CREDITS = 12        # R1
BASE_MAX_CREDITS = 18        # R1
OVERLOAD_GPA = 3.50          # R2
OVERLOAD_MAX_CREDITS = 21    # R2
PROBATION_GPA = 2.00         # R3 (GPA strictly below this = probation)
PROBATION_MIN_CREDITS = 9    # R3
PROBATION_MAX_CREDITS = 12   # R3

ALL_RULE_IDS = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9")


@lru_cache(maxsize=1)
def _default_catalog() -> dict:
    return load_courses()


def _catalog(courses):
    return courses if courses is not None else _default_catalog()


def _norm(code) -> str:
    return str(code).strip().upper()


def _fmt(meeting) -> str:
    day, start, end = meeting
    return f"{day} {start // 60:02d}:{start % 60:02d}-{end // 60:02d}:{end % 60:02d}"


# ------------------------------------------------------------------ R1-R3
def credit_limits(gpa) -> dict:
    """Credit-hour limits for a GPA (R1, R2, R3).  gpa=None -> base limits, status 'unknown'."""
    if gpa is None:
        return {"min": BASE_MIN_CREDITS, "max": BASE_MAX_CREDITS, "rule": "R1", "status": "unknown"}
    if gpa < PROBATION_GPA:
        return {"min": PROBATION_MIN_CREDITS, "max": PROBATION_MAX_CREDITS, "rule": "R3", "status": "probation"}
    if gpa >= OVERLOAD_GPA:
        return {"min": BASE_MIN_CREDITS, "max": OVERLOAD_MAX_CREDITS, "rule": "R2", "status": "overload"}
    return {"min": BASE_MIN_CREDITS, "max": BASE_MAX_CREDITS, "rule": "R1", "status": "standard"}


# --------------------------------------------------------------------- R7
def slot_conflicts(a: Course, b: Course) -> list:
    """Descriptions of every overlapping meeting between two courses ([] = no clash).
    A meeting that ends exactly when another starts is not a clash."""
    found = []
    for ma in a.meetings():
        for mb in b.meetings():
            if ma[0] == mb[0] and ma[1] < mb[2] and mb[1] < ma[2]:
                found.append(f"{a.code} {_fmt(ma)} overlaps {b.code} {_fmt(mb)}")
    return found


def find_clashes(codes, courses=None) -> list:
    """All clashing pairs inside a list of course codes: [(code_a, code_b, [details])]."""
    courses = _catalog(courses)
    codes = [c for c in (_norm(c) for c in codes) if c in courses]
    out = []
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            details = slot_conflicts(courses[a], courses[b])
            if details:
                out.append((a, b, details))
    return out


def catalog_clashes(semester, courses=None) -> list:
    """All clashing pairs among the courses offered in one semester."""
    courses = _catalog(courses)
    return find_clashes([c for c, x in courses.items() if semester in x.offered], courses)


# ----------------------------------------------------------- single course
def check_course(profile, code, courses=None) -> dict:
    """Check the per-course rules R8, R4, R6, R5 for one course.

    Returns {"eligible": bool, "reasons": [...], "rule_refs": [...], "missing": [...],
             "flags": [...]}.  All violated rules are reported, not just the first.
    Plan-level rules (R1-R3 credit load, R7 time clashes) are applied by check_plan().
    """
    courses = _catalog(courses)
    code = _norm(code)
    reasons, refs, missing, flags = [], [], [], []

    def add_ref(r):
        if r not in refs:
            refs.append(r)

    def cannot_verify(text):
        msg = "R9: " + text
        reasons.append(msg)
        flags.append(msg)
        add_ref("R9")

    if code not in courses:
        cannot_verify(f"course code {code} is not in the catalog, so eligibility cannot be verified")
        return {"eligible": False, "reasons": reasons, "rule_refs": refs, "missing": missing, "flags": flags}

    course = courses[code]
    passed = profile.passed

    # R8 - no retaking passed courses
    if passed is not None and code in passed:
        reasons.append(f"R8: {code} is already in the student's passed courses")
        add_ref("R8")

    # R4 - level eligibility
    if profile.level is None:
        cannot_verify(f"student level is missing, so level eligibility (R4) of {code} cannot be verified")
    elif course.level > profile.level:
        reasons.append(f"R4: {code} is a level {course.level} course but the student is in level {profile.level}")
        add_ref("R4")

    # R6 - offered in the student's semester
    if profile.semester is None:
        cannot_verify(f"student semester is missing, so availability (R6) of {code} cannot be verified")
    elif profile.semester not in course.offered:
        reasons.append(f"R6: {code} is not offered in {profile.semester} (offered: {', '.join(course.offered)})")
        add_ref("R6")

    # R5 - prerequisites
    if course.prerequisites:
        if passed is None:
            cannot_verify(f"passed courses are missing, so the prerequisites (R5) of {code} cannot be verified")
        else:
            missing = [p for p in course.prerequisites if p not in passed]
            if missing:
                reasons.append(f"R5: prerequisite(s) not passed: {', '.join(missing)} (required for {code})")
                add_ref("R5")

    return {"eligible": not reasons, "reasons": reasons, "rule_refs": refs, "missing": missing, "flags": flags}


def eligible_courses(profile, courses=None) -> list:
    """Codes of all catalog courses that pass the per-course rules (R8, R4, R6, R5)."""
    courses = _catalog(courses)
    return [c for c in sorted(courses) if check_course(profile, c, courses)["eligible"]]


# ---------------------------------------------------------------- whole plan
def check_plan(profile, codes, courses=None) -> dict:
    """Check a proposed list of courses as a whole.

    Courses are processed in the order given, so an earlier course has priority
    over a later one that clashes with it or that would exceed the credit maximum.
    Returns {"accepted": [codes], "rejected": [{"code", "reasons", "rule_refs", "missing"}],
             "total_credits": int, "flags": [...], "credit_limits": {...}}.
    """
    courses = _catalog(courses)
    limits = credit_limits(profile.gpa)
    accepted, rejected, flags = [], [], list(profile.flags)
    total, seen = 0, set()

    for raw in codes:
        code = _norm(raw)
        if code in seen:
            flags.append(f"NOTE: {code} was listed more than once; the duplicate was ignored")
            continue
        seen.add(code)

        base = check_course(profile, code, courses)
        reasons, refs, missing = list(base["reasons"]), list(base["rule_refs"]), list(base["missing"])

        if base["eligible"]:
            course = courses[code]
            # R7 - time clashes with courses already accepted
            for other in accepted:
                details = slot_conflicts(course, courses[other])
                if details:
                    reasons.append(f"R7: time clash with {other} ({'; '.join(details)})")
                    if "R7" not in refs:
                        refs.append("R7")
            # R1 / R2 / R3 - credit maximum
            new_total = total + course.credits
            if new_total > limits["max"]:
                rule = limits["rule"]
                if rule == "R3":
                    why = f"above the probation maximum of {limits['max']}"
                elif rule == "R2":
                    why = f"above the overload maximum of {limits['max']}"
                else:
                    why = f"above the maximum of {limits['max']}"
                    if limits["status"] == "unknown":
                        why += " (GPA is unknown, so only the base limit was applied)"
                    else:
                        why += f" (overload to {OVERLOAD_MAX_CREDITS} needs GPA >= {OVERLOAD_GPA:.2f}, R2)"
                reasons.append(f"{rule}: adding {code} ({course.credits} cr) would bring the load to {new_total} credit hours, {why}")
                if rule not in refs:
                    refs.append(rule)

        if reasons:
            rejected.append({"code": code, "reasons": reasons, "rule_refs": refs, "missing": missing})
        else:
            accepted.append(code)
            total += courses[code].credits

    # plan-level flags -------------------------------------------------
    min_rule = "R3" if limits["status"] == "probation" else "R1"
    if total < limits["min"]:
        flags.append(f"{min_rule}: total of {total} credit hours is below the minimum of {limits['min']}")
    wanted = profile.desired_credits
    if wanted is not None:
        if wanted > limits["max"]:
            extra = ""
            if limits["rule"] == "R1" and limits["status"] != "unknown":
                extra = f" (overload to {OVERLOAD_MAX_CREDITS} needs GPA >= {OVERLOAD_GPA:.2f}, R2)"
            flags.append(f"{limits['rule']}: desired credit load of {wanted} exceeds the maximum of {limits['max']} credit hours allowed{extra}")
        elif wanted < limits["min"]:
            flags.append(f"{min_rule}: desired credit load of {wanted} is below the minimum of {limits['min']} credit hours")

    unique_flags = list(dict.fromkeys(flags))
    return {"accepted": accepted, "rejected": rejected, "total_credits": total,
            "flags": unique_flags, "credit_limits": limits}


# ------------------------------------------------------- traceability + output
def verify_rule_references(rules_path=None) -> list:
    """Rule IDs used by this module that are missing from registration_rules.md ([] = all traceable)."""
    documented = load_rules(rules_path)
    return [r for r in ALL_RULE_IDS if r not in documented]


def format_plan_result(result: dict, courses=None) -> str:
    """Readable text version of a check_plan() result."""
    courses = _catalog(courses)
    lim = result["credit_limits"]
    lines = [f"Credit limits: {lim['min']}-{lim['max']} ({lim['rule']}, status: {lim['status']})"]
    lines.append(f"ACCEPTED ({result['total_credits']} credit hours):")
    for c in result["accepted"]:
        lines.append(f"  + {c}  {courses[c].title} ({courses[c].credits} cr)")
    if not result["accepted"]:
        lines.append("  (none)")
    lines.append("REJECTED:")
    for r in result["rejected"]:
        title = courses[r["code"]].title if r["code"] in courses else "unknown course"
        lines.append(f"  - {r['code']}  {title}  [{', '.join(r['rule_refs'])}]")
        for reason in r["reasons"]:
            lines.append(f"        {reason}")
    if not result["rejected"]:
        lines.append("  (none)")
    lines.append("FLAGS:")
    for f in result["flags"]:
        lines.append(f"  ! {f}")
    if not result["flags"]:
        lines.append("  (none)")
    return "\n".join(lines)
