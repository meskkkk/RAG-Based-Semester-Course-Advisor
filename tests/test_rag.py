"""Tests for the RAG pipeline and chatbot (Member B).

Run:  python tests/test_rag.py      or      pytest tests/test_rag.py
They use the TF-IDF embedder and no LLM key, so they run offline and fast.
"""
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("ADVISOR_EMBEDDER", "tfidf")
for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER"):
    os.environ.pop(k, None)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunker import build_chunks, chunk_text          # noqa: E402
from retriever import retrieve, rank_courses          # noqa: E402
from recommender import recommend                     # noqa: E402
from chatbot import ChatSession, load_demo_profiles   # noqa: E402

PROFILES = {p.id: p for p in load_demo_profiles(ROOT / "profiles").values()}
EXPECTED = json.loads((ROOT / "profiles" / "expected_results.json").read_text())


def test_chunks_cover_every_course_and_rule():
    chunks = build_chunks()
    courses = json.loads((ROOT / "knowledge_base" / "courses.json").read_text())
    refs = {c.ref for c in chunks}
    assert {c["code"] for c in courses} <= refs
    assert {f"R{i}" for i in range(1, 10)} <= refs
    assert all(c.source and c.kind in ("course", "rule", "terms") for c in chunks)


def test_chunk_text_splits_long_text_with_overlap():
    words = [f"w{i}" for i in range(500)]
    parts = chunk_text(" ".join(words), max_words=200, overlap=40)
    assert len(parts) == 3 and parts[1].split()[0] == "w160"


def test_retrieval_finds_the_right_passage():
    assert retrieve("computer vision images", k=1)[0]["source"] == "course_descriptions.md#CS304"
    top3 = [h["source"] for h in retrieve("academic probation low GPA", k=3)]
    assert "registration_rules.md#R3" in top3
    assert retrieve("time clash overlap", k=1, kind="rule")[0]["source"] == "registration_rules.md#R7"


def test_ranking_skips_passed_courses():
    p = PROFILES["S01"]
    codes = [c for c, _ in rank_courses(p)]
    assert not set(codes) & set(p.passed)
    assert codes[0] == "CS304"     # interest: computer vision


def test_requested_plans_match_expected_results():
    for e in EXPECTED:
        rec = recommend(PROFILES[e["profile"]], requested=e["candidate_plan"])
        assert [c["code"] for c in rec["courses"]] == e["accepted"], e["profile"]
        assert rec["total_credits"] == e["total_credits"]
        assert sorted(x["code"] for x in rec["excluded"]) == sorted(e["rejected"])


def test_every_recommendation_has_reasons_and_sources():
    for p in PROFILES.values():
        rec = recommend(p)
        for c in rec["courses"]:
            assert c["why"] and c["eligibility"] and c["sources"]
            assert any(s.startswith("course_descriptions.md#") for s in c["sources"])
        for x in rec["excluded"]:
            assert x["reasons"]
        assert rec["retrieved"] and rec["summary"]


def test_auto_plan_respects_rules_and_credit_limit():
    for p in PROFILES.values():
        rec = recommend(p)
        assert rec["total_credits"] <= rec["credit_limits"]["max"]
    assert recommend(PROFILES["S06"])["courses"] == []        # missing level: nothing guessed


def test_chatbot_eligibility_add_drop():
    s = ChatSession(PROFILES["S02"])
    assert "No, you cannot take CS301" in s.ask("can I take CS301?") and "MATH202" in s.ask("can I take CS301?")
    s.ask("I passed MATH202")
    assert "added" in s.ask("add CS301")
    assert "CS301" in s.ask("show my plan")
    assert "Removed CS301" in s.ask("drop CS301")
    assert "Sources:" in s.ask("what is probation?")
    assert PROFILES["S02"].passed.count("MATH202") == 0        # original profile unchanged


def test_new_student_interview_builds_a_valid_profile():
    s = ChatSession.new_student()
    for msg in ["Mona", "2", "spring", "CS101, MATH101, XYZ999", "3.7", "python", "web, data", "15"]:
        reply = s.ask(msg)
    p = s.profile
    assert (p.name, p.level, p.semester, p.gpa, p.desired_credits) == ("Mona", 2, "Spring", 3.7, 15)
    assert p.passed == ["CS101", "MATH101"] and p.interests == ["web", "data"]
    assert not s.interviewing and "Sources:" in reply
    total = sum(s.courses[c].credits for c in s.plan)
    assert total <= 21
    assert "Level" in s.ask("list courses") and "BM" in s.ask("who are you?")


def test_interview_rejects_bad_answers_without_guessing():
    s = ChatSession.new_student()
    s.ask("Omar")
    assert "1" in s.ask("I am not sure") and s.profile.level is None    # asks again
    s.ask("1"); s.ask("fall")
    assert "course codes" in s.ask("some programming") and s.profile.passed is None


def run_all(verbose=True):
    passed, failed = [], []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed.append(name)
            except Exception:
                failed.append((name, traceback.format_exc()))
    if verbose:
        for n in passed:
            print("PASS", n)
        for n, tb in failed:
            print("FAIL", n, "\n", tb)
        print(f"\n{len(passed)} passed, {len(failed)} failed")
    return passed, failed


if __name__ == "__main__":
    _, f = run_all()
    sys.exit(1 if f else 0)
