"""Tests for the knowledge base, profile module and rule checker (task A7).

Run with:  pytest tests/            or             python tests/test_rules.py
One pass case and one fail case per rule (R1-R9), plus KB validation, profile
validation and a comparison of all six demo profiles with expected_results.json.
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import kb_loader                                     # noqa: E402
from kb_loader import load_courses, validate_course_records, validate_kb, KBValidationError  # noqa: E402
from profile import Profile, load_profile, prompt_profile   # noqa: E402
import rules                                          # noqa: E402
from rules import check_course, check_plan, credit_limits    # noqa: E402

COURSES = load_courses()
KB = ROOT / "knowledge_base"
PROFILES = ROOT / "profiles"

BASE_PASSED = ["CS101", "CS102", "MATH101", "PHYS101", "ENG101", "MATH102",
               "CS201", "MATH201", "MATH202", "CS202", "CS203", "CS204"]


def make_profile(**kw):
    """A valid level-3 Fall profile; override any field. Passing a field as None makes it missing."""
    data = dict(id="T", name="Test", level=3, gpa=3.0, passed=list(BASE_PASSED), skills=[],
                interests=[], desired_credits=15, semester="Fall")
    data.update(kw)
    return Profile.from_dict(data, COURSES)


# ============================================================ A1-A3 knowledge base
def test_kb_is_valid_and_meets_size_requirements():
    summary = validate_kb()
    assert 25 <= summary["courses"] <= 30
    assert summary["descriptions"] == summary["courses"]
    levels = {c.level for c in COURSES.values()}
    assert levels == {1, 2, 3, 4}


def test_kb_has_chained_prerequisites_and_clashes():
    # chain CS101 -> CS102 -> CS201 -> CS301 -> CS401
    chain = ["CS101", "CS102", "CS201", "CS301", "CS401"]
    for a, b in zip(chain, chain[1:]):
        assert a in COURSES[b].prerequisites
    assert rules.catalog_clashes("Fall"), "catalog needs a clashing pair in Fall"
    assert rules.catalog_clashes("Spring"), "catalog needs a clashing pair in Spring"


def test_kb_descriptions_have_4_to_6_sentences():
    for code, text in kb_loader.load_descriptions().items():
        assert 4 <= kb_loader.count_sentences(text) <= 6, code


def test_every_rule_id_used_by_checker_is_documented():
    assert rules.verify_rule_references() == []


def test_credit_numbers_in_code_appear_in_rules_document():
    doc = kb_loader.load_rules()
    assert "12" in doc["R1"]["text"] and "18" in doc["R1"]["text"]
    assert "3.50" in doc["R2"]["text"] and "21" in doc["R2"]["text"]
    assert "2.00" in doc["R3"]["text"] and "9" in doc["R3"]["text"] and "12" in doc["R3"]["text"]
    assert (rules.BASE_MIN_CREDITS, rules.BASE_MAX_CREDITS) == (12, 18)
    assert (rules.OVERLOAD_GPA, rules.OVERLOAD_MAX_CREDITS) == (3.5, 21)
    assert (rules.PROBATION_GPA, rules.PROBATION_MIN_CREDITS, rules.PROBATION_MAX_CREDITS) == (2.0, 9, 12)


def test_rules_document_is_labelled_fictional():
    assert "FICTIONAL" in (KB / "registration_rules.md").read_text(encoding="utf-8")
    assert "FICTIONAL" in (KB / "course_descriptions.md").read_text(encoding="utf-8")


def test_validator_catches_bad_catalogs():
    good = json.loads((KB / "courses.json").read_text(encoding="utf-8"))
    assert validate_course_records(good) == []

    dup = copy.deepcopy(good) + [copy.deepcopy(good[0])]
    assert any("duplicate" in p for p in validate_course_records(dup))

    bad_prereq = copy.deepcopy(good)
    bad_prereq[8]["prerequisites"].append("NOPE999")
    assert any("NOPE999" in p for p in validate_course_records(bad_prereq))

    missing_field = copy.deepcopy(good)
    del missing_field[0]["credits"]
    assert any("credits" in p for p in validate_course_records(missing_field))

    bad_slot = copy.deepcopy(good)
    bad_slot[0]["timeslot"] = ["Sunday 9am"]
    assert any("timeslot" in p for p in validate_course_records(bad_slot))

    backwards = copy.deepcopy(good)
    backwards[0]["timeslot"] = ["Sun 10:00-09:00"]
    assert any("end after it starts" in p for p in validate_course_records(backwards))

    cycle = copy.deepcopy(good)
    cycle[6]["prerequisites"] = ["CS301"]       # CS201 now needs CS301, which needs CS201
    assert any("cycle" in p for p in validate_course_records(cycle))

    bad_semester = copy.deepcopy(good)
    bad_semester[0]["offered"] = ["Summer"]
    assert any("offered" in p for p in validate_course_records(bad_semester))


def test_load_courses_raises_on_invalid_catalog(tmp_path):
    bad = [{"code": "CS101", "title": "x"}]
    p = tmp_path / "courses.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    try:
        load_courses(p)
    except KBValidationError as exc:
        assert exc.problems
    else:
        raise AssertionError("expected KBValidationError")


# ============================================================ A4 profile module
def test_profile_valid_has_no_flags():
    assert make_profile().flags == []


def test_profile_missing_level_and_credits_are_flagged():
    p = make_profile(level=None, desired_credits=None)
    assert p.level is None and p.desired_credits is None
    assert any("'level'" in f for f in p.flags)
    assert any("'desired_credits'" in f for f in p.flags)


def test_profile_unknown_course_code_is_flagged():
    p = make_profile(passed=["CS101", "ZZZ999"])
    assert any("ZZZ999" in f for f in p.flags)


def test_profile_invalid_values_are_flagged_not_guessed():
    p = make_profile(level=7, gpa=5.2, semester="Summer", desired_credits=-3)
    assert p.level is None and p.gpa is None and p.semester is None and p.desired_credits is None
    assert len(p.flags) == 4


def test_profile_empty_passed_list_is_valid_but_missing_is_not():
    assert make_profile(passed=[]).flags == []
    assert any("'passed'" in f for f in make_profile(passed=None).flags)


def test_profile_normalises_codes_and_semester():
    p = make_profile(passed=[" cs101 ", "CS101", "cs102"], semester="fall")
    assert p.passed == ["CS101", "CS102"] and p.semester == "Fall"


def test_load_profile_from_file_and_roundtrip():
    p = load_profile(PROFILES / "student_01.json", COURSES)
    assert p.id == "S01" and p.level == 3 and p.flags == []
    assert Profile.from_dict(p.to_dict(), COURSES).to_dict() == p.to_dict()


def test_interactive_prompt_complete_and_partial():
    answers = iter(["S77", "Mona", "2", "3.1", "CS101, MATH101", "python", "AI, web", "15", "Fall"])
    lines = []
    p = prompt_profile(input_fn=lambda _: next(answers), print_fn=lines.append, courses=COURSES)
    assert p.level == 2 and p.gpa == 3.1 and p.passed == ["CS101", "MATH101"] and p.flags == []

    blank = iter([""] * 9)
    p2 = prompt_profile(input_fn=lambda _: next(blank), print_fn=lines.append, courses=COURSES)
    assert p2.level is None and p2.passed is None and len(p2.flags) == 5   # level, gpa, passed, credits, semester
    new_student = iter(["", "", "1", "", "none", "", "", "15", "Fall"])
    p3 = prompt_profile(input_fn=lambda _: next(new_student), print_fn=lines.append, courses=COURSES)
    assert p3.passed == []


# ============================================================ A6/A7 rules, one pass + one fail each
def test_R1_credit_range_pass_and_fail():
    p = make_profile(gpa=3.0)
    ok = check_plan(p, ["CS301", "CS302", "CS205", "CS307"])           # 13 credits, inside 12-18
    assert ok["total_credits"] == 13 and not any(f.startswith("R1:") for f in ok["flags"])
    low = check_plan(p, ["CS301"])                                       # 3 credits, below the minimum
    assert any(f.startswith("R1:") and "below the minimum of 12" in f for f in low["flags"])


def test_R1_maximum_rejects_course_with_rule_id():
    p = load_profile(PROFILES / "student_04.json", COURSES)
    r = check_plan(p, ["CS402", "CS404", "CS490", "CS307", "CS304", "HUM401", "CS204"])
    assert r["total_credits"] == 18
    rej = {x["code"]: x for x in r["rejected"]}
    assert rej["CS204"]["rule_refs"] == ["R1"]
    assert "21" in rej["CS204"]["reasons"][0] and "18" in rej["CS204"]["reasons"][0]


def test_R2_overload_pass_and_fail():
    plan = ["CS402", "CS404", "CS490", "CS307", "CS304", "HUM401", "CS204"]   # 21 credits in total
    passed = BASE_PASSED + ["CS205", "CS301", "CS302", "CS303", "CS305", "CS306", "CS310"]
    passed.remove("CS204")
    high = make_profile(level=4, gpa=3.5, passed=passed)        # exactly 3.50 -> overload allowed
    r = check_plan(high, plan)
    assert r["total_credits"] == 21 and r["rejected"] == [] and r["credit_limits"]["rule"] == "R2"
    almost = make_profile(level=4, gpa=3.49, passed=passed)     # just below -> no overload
    r = check_plan(almost, plan)
    assert r["total_credits"] == 18 and r["rejected"][0]["rule_refs"] == ["R1"]
    # even with overload, 21 is the ceiling
    assert credit_limits(4.0)["max"] == 21


def test_R3_probation_pass_and_fail():
    prob = make_profile(gpa=1.99)
    r = check_plan(prob, ["CS301", "CS302", "CS205", "CS307"])           # 3+3+3+4
    assert r["accepted"] == ["CS301", "CS302", "CS205"] and r["total_credits"] == 9
    assert r["rejected"][0]["code"] == "CS307" and r["rejected"][0]["rule_refs"] == ["R3"]
    assert not any("below the minimum" in f for f in r["flags"])          # 9 is the probation minimum
    boundary = make_profile(gpa=2.00)                                     # exactly 2.00 is NOT probation
    assert credit_limits(boundary.gpa)["rule"] == "R1"
    below_min = check_plan(prob, ["CS301", "CS302"])                      # 6 < 9
    assert any(f.startswith("R3:") and "below the minimum of 9" in f for f in below_min["flags"])


def test_R4_level_pass_and_fail():
    p = make_profile(level=3)
    assert check_course(p, "CS301")["eligible"] is True                   # level 3 course, level 3 student
    res = check_course(p, "CS402")                                        # level 4 course
    assert res["eligible"] is False and "R4" in res["rule_refs"]
    assert check_course(make_profile(level=1, passed=[]), "CS101")["eligible"] is True   # lower level is fine


def test_R5_prerequisites_pass_and_fail():
    ok = check_course(make_profile(), "CS301")
    assert ok["eligible"] and ok["missing"] == []
    p = make_profile(passed=[c for c in BASE_PASSED if c != "MATH202"])
    res = check_course(p, "CS301")
    assert not res["eligible"] and res["rule_refs"] == ["R5"] and res["missing"] == ["MATH202"]
    # chained prerequisites: CS401 needs CS301 which the student has not passed
    chain = check_course(make_profile(level=4), "CS401", COURSES)
    assert chain["missing"] == ["CS301"]


def test_R6_offered_semester_pass_and_fail():
    assert check_course(make_profile(semester="Fall"), "CS301")["eligible"]
    res = check_course(make_profile(semester="Spring"), "CS301")           # CS301 is a Fall course
    assert not res["eligible"] and res["rule_refs"] == ["R6"]
    assert check_course(make_profile(semester="Spring", passed=BASE_PASSED), "PHYS101")["rule_refs"] == ["R8"]  # offered both, only R8 fails


def test_R7_time_clash_pass_and_fail():
    p = make_profile()
    r = check_plan(p, ["CS301", "CS304"])                                  # identical slots
    assert r["accepted"] == ["CS301"] and r["rejected"][0]["rule_refs"] == ["R7"]
    # partial overlap (Tue 12:30-14:00 vs Tue 13:30-15:00)
    sp = make_profile(level=4, semester="Spring", passed=BASE_PASSED + ["CS205", "CS301"])
    r = check_plan(sp, ["CS305", "CS306"])
    assert r["accepted"] == ["CS305"] and r["rejected"][0]["rule_refs"] == ["R7"]
    # back-to-back is allowed: CS205 ends 12:30, CS301 starts 12:30
    r = check_plan(p, ["CS205", "CS301"])
    assert r["accepted"] == ["CS205", "CS301"] and r["rejected"] == []
    # earlier course has priority: reversing the order flips which one is kept
    assert check_plan(p, ["CS304", "CS301"])["accepted"] == ["CS304"]


def test_R8_no_retake_pass_and_fail():
    p = make_profile()
    assert check_course(p, "CS301")["eligible"]
    res = check_course(p, "CS204")                                         # already passed
    assert not res["eligible"] and "R8" in res["rule_refs"]


def test_R9_missing_data_is_flagged_never_guessed():
    # missing level -> nothing is approved, flag raised
    p = make_profile(level=None)
    res = check_course(p, "CS302")          # prerequisite CS202 is passed, so only the missing level blocks it
    assert res["eligible"] is False and res["rule_refs"] == ["R9"] and res["flags"]
    # missing semester
    res = check_course(make_profile(semester=None), "CS301")
    assert res["eligible"] is False and "R9" in res["rule_refs"]
    # missing passed list: a course WITH prerequisites cannot be verified, one without can
    p = make_profile(passed=None)
    assert check_course(p, "CS301")["eligible"] is False
    res = check_course(p, "ENG101")                                        # no prerequisites, so R5 is not needed
    assert res["missing"] == [] and "R5" not in res["rule_refs"]
    # unknown course code
    res = check_course(make_profile(), "NOPE123")
    assert res["eligible"] is False and res["rule_refs"] == ["R9"]
    # missing GPA: base limits applied, flag raised in the plan
    r = check_plan(make_profile(gpa=None), ["CS301", "CS302", "CS205", "CS307"])
    assert r["credit_limits"]["status"] == "unknown"
    assert any("'gpa'" in f for f in r["flags"])
    # control: complete data -> no R9 anywhere
    assert "R9" not in check_course(make_profile(), "CS301")["rule_refs"]


def test_check_plan_contract_shape():
    r = check_plan(make_profile(), ["CS301", "CS999"])
    assert set(["accepted", "rejected", "total_credits", "flags"]) <= set(r)
    assert isinstance(r["accepted"], list) and isinstance(r["total_credits"], int)
    c = check_course(make_profile(), "CS301")
    assert set(["eligible", "reasons", "rule_refs", "missing"]) <= set(c)
    assert all(set(["code", "reasons", "rule_refs", "missing"]) <= set(x) for x in r["rejected"])


def test_duplicates_and_lowercase_codes_are_handled():
    r = check_plan(make_profile(), ["cs301", "CS301", " cs302 "])
    assert r["accepted"] == ["CS301", "CS302"]
    assert any("more than once" in f for f in r["flags"])


def test_every_rejection_has_a_documented_rule_id():
    documented = set(kb_loader.load_rules())
    for f in sorted(PROFILES.glob("student_*.json")):
        p = load_profile(f, COURSES)
        for rej in check_plan(p, list(COURSES))["rejected"]:
            assert rej["reasons"] and rej["rule_refs"]
            assert set(rej["rule_refs"]) <= documented


def test_eligible_courses_matches_check_course():
    p = make_profile()
    elig = rules.eligible_courses(p)
    for code in COURSES:
        assert (code in elig) == check_course(p, code)["eligible"]
    assert "CS301" in elig and "CS204" not in elig and "CS402" not in elig


# ============================================================ six demo profiles vs expected
def _expected():
    return json.loads((PROFILES / "expected_results.json").read_text(encoding="utf-8"))


def test_six_profiles_exist_and_cover_required_cases():
    exp = _expected()
    assert len(exp) == 6 and len(list(PROFILES.glob("student_*.json"))) == 6
    assert any(any("R5" in r["rule_refs"] for r in e["rejected"].values()) for e in exp)      # unmet prerequisite
    assert any(any("R1" in r["rule_refs"] for r in e["rejected"].values()) for e in exp)      # credit limit
    assert any(any("R7" in r["rule_refs"] for r in e["rejected"].values()) for e in exp)      # time clash
    assert any("R3" in e["credit_limit"]["rule"] for e in exp)                                # low GPA
    assert any("R9" in e["flag_rules"] for e in exp)                                          # missing data
    assert any(not e["rejected"] for e in exp)                                                # normal strong student


def test_all_six_profiles_match_expected_results():
    for e in _expected():
        p = load_profile(PROFILES / e["file"], COURSES)
        r = check_plan(p, e["candidate_plan"])
        tag = e["profile"]
        assert r["accepted"] == e["accepted"], f"{tag}: accepted {r['accepted']} != {e['accepted']}"
        assert r["total_credits"] == e["total_credits"], tag
        got_rej = {x["code"]: x for x in r["rejected"]}
        assert set(got_rej) == set(e["rejected"]), f"{tag}: rejected {sorted(got_rej)}"
        for code, want in e["rejected"].items():
            assert got_rej[code]["rule_refs"] == want["rule_refs"], f"{tag} {code}"
            assert got_rej[code]["missing"] == want["missing"], f"{tag} {code}"
        cl = r["credit_limits"]
        assert (cl["min"], cl["max"], cl["rule"]) == (e["credit_limit"]["min"], e["credit_limit"]["max"], e["credit_limit"]["rule"]), tag
        assert sorted(f.split(":")[0] for f in r["flags"]) == sorted(e["flag_rules"]), f"{tag}: flags {r['flags']}"
        for needle in e["flag_contains"]:
            assert any(needle in f for f in r["flags"]), f"{tag}: no flag containing {needle!r}"


# ------------------------------------------------------------------ runner
def run_all(verbose=True):
    """Run every test_* function without pytest. Returns (passed, failed) lists."""
    import tempfile, traceback, inspect
    passed, failed = [], []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in inspect.signature(fn).parameters:
                    with tempfile.TemporaryDirectory() as d:
                        fn(Path(d))
                else:
                    fn()
                passed.append(name)
            except Exception:
                failed.append((name, traceback.format_exc()))
    if verbose:
        for n in passed:
            print("PASS", n)
        for n, tb in failed:
            print("FAIL", n)
            print(tb)
        print(f"\n{len(passed)} passed, {len(failed)} failed")
    return passed, failed


if __name__ == "__main__":
    _, bad = run_all()
    sys.exit(1 if bad else 0)
