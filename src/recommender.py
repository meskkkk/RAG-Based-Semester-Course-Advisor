"""Recommendation pipeline and output (task B5).

    retrieve  ->  rule check (Member A)  ->  generate explanation  ->  formatted output

Two modes:
  * auto      : the advisor builds the plan.  Courses are tried in order of relevance
                (retrieval score) and kept only if Member A's ``check_plan`` accepts them,
                until the student's desired credit load is reached (never above the
                allowed maximum).
  * requested : the student names the courses they want; the plan is checked as given
                (this is what the six demo profiles use, so the unmet-prerequisite and
                credit-limit cases are visible).

Public API (shared interface contract):
    recommend(profile, requested=None, llm=None) ->
        {"courses", "excluded", "total_credits", "sources", ...}
    format_recommendation(rec) -> str
"""
from __future__ import annotations

import re

from kb_loader import load_courses
from rules import check_course, check_plan, credit_limits
from retriever import rank_courses, course_evidence, rule_evidence, retrieve, profile_query
from generator import SYSTEM_PROMPT, build_recommendation_prompt, get_llm

N_EXCLUDED_SHOWN = 6        # how many relevant-but-excluded courses to explain in auto mode
N_RETRIEVED_SHOWN = 6       # top passages shown as "retrieved evidence"

_COURSES = None


def _courses():
    global _COURSES
    if _COURSES is None:
        _COURSES = load_courses()
    return _COURSES


# ------------------------------------------------------------ explanations
def _why(profile, code, score=None) -> str:
    """Why the course fits the student: matched interests/skills, else relevance score."""
    c = _courses()[code]
    hay = " ".join([c.title, c.description, " ".join(c.tags)]).lower()
    hits = [x for x in list(profile.interests or []) + list(profile.skills or [])
            if re.search(r"\b" + re.escape(x.lower()) + r"\b", hay)]
    if hits:
        text = "matches your " + ", ".join(dict.fromkeys(hits)) + f" (topics: {', '.join(c.tags)})"
    else:
        text = f"broadens your study plan (topics: {', '.join(c.tags)})"
    if score is not None:
        text += f"; relevance {score:.2f}"
    return text


def _eligibility(profile, code, limits, plan_codes) -> tuple:
    """Plain-language eligibility explanation for an accepted course + the rule IDs it relied on."""
    c = _courses()[code]
    parts = [f"level {c.level} <= your level {profile.level} (R4)"]
    refs = ["R4"]
    if c.prerequisites:
        parts.append(f"prerequisites {', '.join(c.prerequisites)} passed (R5)")
    else:
        parts.append("no prerequisites (R5)")
    refs.append("R5")
    parts.append(f"offered in {profile.semester} (R6)")
    refs.append("R6")
    if len(plan_codes) > 1:
        parts.append("no time clash with the other courses (R7)")
        refs.append("R7")
    parts.append(f"fits the {limits['min']}-{limits['max']} credit limit ({limits['rule']})")
    refs.append(limits["rule"])
    return "; ".join(parts), refs


# ------------------------------------------------------------------ planning
def _target_credits(profile, limits) -> int:
    wanted = profile.desired_credits
    if wanted is None:
        return limits["min"]
    return max(limits["min"], min(wanted, limits["max"]))


def _auto_plan(profile, ranked, limits):
    """Greedy: take the most relevant course that the rule checker accepts, until the target is met."""
    courses = _courses()
    target = _target_credits(profile, limits)
    plan, total = [], 0
    for code, _ in ranked:
        if total >= target:
            break
        cr = courses[code].credits
        if total + cr > limits["max"]:
            continue                                   # would break the credit maximum
        if total >= limits["min"] and total + cr > target + 1:
            continue                                   # minimum already met: avoid overshooting the wish by 2+
        if code in check_plan(profile, plan + [code])["accepted"]:
            plan.append(code)
            total += cr
    return plan, target


# ------------------------------------------------------------------- public
def recommend(profile, requested=None, llm=None) -> dict:
    """Full RAG recommendation for one student. See the module docstring for the two modes."""
    courses = _courses()
    llm = llm or get_llm()
    limits = credit_limits(profile.gpa)
    ranked = rank_courses(profile)
    score_of = dict(ranked)

    if requested:
        mode = "requested"
        plan_codes = [str(c).strip().upper() for c in requested]
        result = check_plan(profile, plan_codes)
        excluded_raw = result["rejected"]
    else:
        mode = "auto"
        plan_codes, target = _auto_plan(profile, ranked, limits)
        result = check_plan(profile, plan_codes)
        # explain the most relevant courses that did NOT make it into the plan
        others = [c for c, _ in ranked if c not in result["accepted"]][:N_EXCLUDED_SHOWN]
        excluded_raw = []
        for code in others:
            single = check_course(profile, code)
            if not single["eligible"]:
                excluded_raw.append({"code": code, "reasons": single["reasons"],
                                     "rule_refs": single["rule_refs"], "missing": single["missing"]})
                continue
            trial = check_plan(profile, result["accepted"] + [code])
            rej = [r for r in trial["rejected"] if r["code"] == code]
            if rej:
                excluded_raw.append(rej[0])
            else:
                excluded_raw.append({"code": code, "reasons": [
                    f"eligible, but not needed: the plan already reaches your target of {target} credit hours"],
                    "rule_refs": [], "missing": []})

    # ---- accepted courses with reasons, eligibility and evidence
    accepted = []
    evidence, sources = [], []

    def add_evidence(items):
        for it in items:
            if it["source"] not in sources:
                sources.append(it["source"])
                evidence.append(it)

    for code in result["accepted"]:
        c = courses[code]
        elig, refs = _eligibility(profile, code, limits, result["accepted"])
        ev = course_evidence(code, score_of.get(code))
        add_evidence(ev)
        add_evidence(rule_evidence(refs))
        accepted.append({"code": code, "title": c.title, "credits": c.credits,
                         "why": _why(profile, code, score_of.get(code)), "eligibility": elig,
                         "sources": [e["source"] for e in ev] + [f"registration_rules.md#{r}" for r in refs]})

    excluded = []
    for r in excluded_raw:
        code = r["code"]
        known = code in courses
        ev = course_evidence(code) if known else []
        add_evidence(ev)
        add_evidence(rule_evidence(r["rule_refs"]))
        excluded.append({"code": code, "title": courses[code].title if known else "(not in catalog)",
                         "reasons": r["reasons"], "rule_refs": r["rule_refs"], "missing": r.get("missing", []),
                         "sources": [e["source"] for e in ev] + [f"registration_rules.md#{x}" for x in r["rule_refs"]]})

    # flag rules also count as evidence
    flag_rules = sorted({f.split(":")[0] for f in result["flags"] if f[:1] == "R" and ":" in f})
    add_evidence(rule_evidence(flag_rules))

    retrieved = retrieve(profile_query(profile), k=N_RETRIEVED_SHOWN)

    rec = {
        "student": profile.id, "name": profile.name, "mode": mode,
        "requested": plan_codes if mode == "requested" else None,
        "query": profile_query(profile),
        "courses": accepted, "excluded": excluded,
        "total_credits": result["total_credits"], "credit_limits": result["credit_limits"],
        "flags": result["flags"], "retrieved": retrieved, "evidence": evidence, "sources": sources,
    }
    text = llm.generate(SYSTEM_PROMPT, build_recommendation_prompt(profile, rec))
    rec["summary"] = text or _template_summary(profile, rec)
    rec["llm"] = llm.name if text else "offline template" + (
        f" (LLM error: {llm.last_error})" if getattr(llm, "last_error", None) else "")
    return rec


def _template_summary(profile, rec) -> str:
    """Answer used when no LLM is available: built only from checked facts and sources."""
    name = (profile.name or "student").split()[0]
    lines = []
    if rec["courses"]:
        interests = ", ".join(profile.interests) if profile.interests else "your study plan"
        lines.append(f"Hi {name}, based on your interest in {interests} and the courses you have passed, "
                     f"these {len(rec['courses'])} course(s) ({rec['total_credits']} credit hours) are allowed "
                     f"for you this {profile.semester or 'semester'}:")
        for c in rec["courses"]:
            lines.append(f"  * {c['code']} {c['title']}: {c['why']} [{c['sources'][0]}]")
    else:
        lines.append(f"Hi {name}, I cannot recommend any course yet.")
    for x in rec["excluded"]:
        cite = " ".join(f"[registration_rules.md#{r}]" for r in x["rule_refs"])
        lines.append(f"  x {x['code']} not included: {'; '.join(x['reasons'])} {cite}".rstrip())
    if rec["flags"]:
        lines.append("Please check: " + " | ".join(rec["flags"]))
    return "\n".join(lines)


# -------------------------------------------------------------------- output
def format_recommendation(rec, show_evidence: bool = True, max_chars: int = 220) -> str:
    """Readable report of a recommend() result."""
    lim = rec["credit_limits"]
    out = [f"STUDENT {rec['student']} - {rec['name']}   (mode: {rec['mode']}, LLM: {rec['llm']})"]
    if rec["requested"]:
        out.append("Requested courses: " + ", ".join(rec["requested"]))
    out.append(f"Search query: {rec['query']}")
    if show_evidence:
        out.append("\nRETRIEVED EVIDENCE (top passages for this student):")
        for p in rec["retrieved"]:
            txt = p["text"].replace("\n", " ")
            out.append(f"  [{p['score']:.3f}] {p['source']}: {txt[:max_chars]}{'...' if len(txt) > max_chars else ''}")
    out.append(f"\nRECOMMENDED COURSES  (total {rec['total_credits']} credit hours; "
               f"allowed {lim['min']}-{lim['max']}, {lim['rule']} {lim['status']})")
    if not rec["courses"]:
        out.append("  (none)")
    for c in rec["courses"]:
        out.append(f"  + {c['code']} {c['title']} ({c['credits']} cr)")
        out.append(f"      why: {c['why']}")
        out.append(f"      eligible: {c['eligibility']}")
        out.append(f"      sources: {', '.join(c['sources'])}")
    out.append("\nEXCLUDED COURSES")
    if not rec["excluded"]:
        out.append("  (none)")
    for x in rec["excluded"]:
        out.append(f"  - {x['code']} {x['title']}  [{', '.join(x['rule_refs']) or 'no rule broken'}]")
        for r in x["reasons"]:
            out.append(f"      {r}")
        if x["sources"]:
            out.append(f"      sources: {', '.join(x['sources'])}")
    out.append("\nFLAGS")
    out += [f"  ! {f}" for f in rec["flags"]] or ["  (none)"]
    out.append("\nADVISOR MESSAGE")
    out.append(rec["summary"])
    out.append("\nALL SOURCES USED: " + ", ".join(rec["sources"]))
    return "\n".join(out)
