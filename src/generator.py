"""LLM connection and prompt design (task B4).

Supported providers (pick one with LLM_PROVIDER, or it is detected from the key you set):

    provider  env key                          default model (override with LLM_MODEL)
    groq      GROQ_API_KEY                     openai/gpt-oss-20b          (free tier)
    gemini    GEMINI_API_KEY or GOOGLE_API_KEY gemini-2.5-flash            (free tier)
    openai    OPENAI_API_KEY                   gpt-4o-mini
    offline   (no key)                         template answers built from the evidence

The LLM never decides whether a course is allowed.  Acceptance and rejection
come from Member A's rule checker; the LLM only explains the decision in plain
language, using the retrieved passages and citing them.  If the API call fails
for any reason, the offline answer is returned instead with a short note, so the
notebook never crashes because of the network.

Public API:
    get_llm(provider=None) -> LLM          .name, .generate(system, user) -> str
    SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT
    build_recommendation_prompt(profile, rec) -> str
    build_question_prompt(question, profile, passages, facts="") -> str
    format_evidence(passages) -> str
"""
from __future__ import annotations

import os

DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-20b",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
}
# tried in order if the default model is retired or unavailable
BACKUP_MODELS = {
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"],
    "gemini": ["gemini-2.0-flash", "gemini-flash-latest"],
    "openai": ["gpt-4.1-mini"],
}

SYSTEM_PROMPT = """You are a university course-registration advisor.
Rules you must follow:
1. Use ONLY the evidence passages and the rule-checker results given to you. Do not invent courses, rules, grades or numbers.
2. The rule checker's ACCEPTED / REJECTED decisions are final. Never recommend a rejected course and never question an accepted one.
3. Cite the source of every fact in square brackets exactly as written, for example [course_descriptions.md#CS301] or [registration_rules.md#R5].
4. If information is missing or flagged, say so clearly and tell the student what to provide. Never guess it.
5. Be concise, friendly and specific to this student's interests, skills and goals."""

CHAT_SYSTEM_PROMPT = SYSTEM_PROMPT + """
6. Your name is BM. You are chatting with the student. Answer in at most 6 sentences, warmly and naturally.
7. If the evidence does not contain the answer, say that the registration documents do not cover it.
8. If the student is just chatting or asks about something outside course registration, reply briefly and kindly,
   then steer the conversation back to their courses."""


# ----------------------------------------------------------------- formatting
def format_evidence(passages) -> str:
    lines = []
    for p in passages:
        score = "" if p.get("score") is None else f" (similarity {p['score']:.2f})"
        lines.append(f"[{p['source']}]{score}\n{p['text']}")
    return "\n\n".join(lines)


def _profile_block(profile) -> str:
    return (f"Student: {profile.name or profile.id or 'unknown'} | level: {profile.level} | GPA: {profile.gpa} | "
            f"semester: {profile.semester} | desired credits: {profile.desired_credits}\n"
            f"Passed courses: {', '.join(profile.passed) if profile.passed else profile.passed}\n"
            f"Skills: {', '.join(profile.skills) or '-'} | Interests/goals: {', '.join(profile.interests) or '-'}")


def build_recommendation_prompt(profile, rec) -> str:
    acc = "\n".join(f"- {c['code']} {c['title']} ({c['credits']} cr). Fit: {c['why']}. Eligibility: {c['eligibility']}"
                    for c in rec["courses"]) or "- none"
    rej = "\n".join(f"- {x['code']} {x['title']}: {'; '.join(x['reasons'])}" for x in rec["excluded"]) or "- none"
    flags = "\n".join(f"- {f}" for f in rec["flags"]) or "- none"
    lim = rec["credit_limits"]
    return f"""{_profile_block(profile)}

RULE-CHECKER RESULT (final, do not change):
Credit limits: {lim['min']}-{lim['max']} credit hours ({lim['rule']}, {lim['status']}). Total recommended: {rec['total_credits']}.
ACCEPTED:
{acc}
EXCLUDED:
{rej}
FLAGS:
{flags}

EVIDENCE PASSAGES:
{format_evidence(rec['evidence'])}

Write the advisor's message to this student:
- one short paragraph that explains why this set of accepted courses suits their interests and level;
- one bullet per accepted course with a one-sentence reason and its citation;
- one bullet per excluded course with the reason and the rule citation;
- if there are flags, a final line telling the student what to fix or provide."""


def build_question_prompt(question, profile, passages, facts: str = "") -> str:
    extra = f"\nRULE-CHECKER FACTS (final):\n{facts}\n" if facts else ""
    return f"""{_profile_block(profile) if profile is not None else 'Student profile: not given'}
{extra}
EVIDENCE PASSAGES:
{format_evidence(passages)}

Student question: {question}"""


# ----------------------------------------------------------------- providers
class OfflineLLM:
    """No model: returns None so callers build a template answer from the evidence."""
    name = "offline (template answers, no LLM key set)"

    def generate(self, system: str, user: str):
        return None


class _APILLM:
    def __init__(self, provider: str, model: str | None = None, temperature: float = 0.2):
        self.provider = provider
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODELS[provider]
        self.temperature = temperature
        self.name = f"{provider}:{self.model}"
        self.last_error = None
        self._client = self._make_client()

    def _make_client(self):
        if self.provider == "groq":
            from groq import Groq
            return Groq(api_key=os.environ["GROQ_API_KEY"])
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        if self.provider == "gemini":
            from google import genai
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            return genai.Client(api_key=key)
        raise ValueError(f"unknown provider {self.provider}")

    def generate(self, system: str, user: str):
        text = self._call(system, user)
        if text is None and self.last_error and "model" in self.last_error.lower():
            for backup in BACKUP_MODELS.get(self.provider, []):      # model retired? try the next one
                old, self.model = self.model, backup
                text = self._call(system, user)
                if text is not None:
                    self.name = f"{self.provider}:{self.model}"
                    print(f"[generator] Model {old} unavailable; switched to {backup}.")
                    return text
        return text

    def _call(self, system: str, user: str):
        self.last_error = None
        try:
            if self.provider in ("groq", "openai"):
                resp = self._client.chat.completions.create(
                    model=self.model, temperature=self.temperature, max_tokens=900,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                return resp.choices[0].message.content.strip()
            if self.provider == "gemini":
                from google.genai import types
                resp = self._client.models.generate_content(
                    model=self.model, contents=user,
                    config=types.GenerateContentConfig(system_instruction=system, temperature=self.temperature,
                                                       max_output_tokens=900))
                return (resp.text or "").strip() or None
        except Exception as exc:          # network, quota, wrong model name ...
            self.last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            print(f"[generator] LLM call failed ({self.last_error}); will try a backup model or use the offline answer.")
            return None


def get_llm(provider: str | None = None, model: str | None = None):
    """Return an LLM client. With no provider given, use LLM_PROVIDER or the first API key found."""
    provider = (provider or os.environ.get("LLM_PROVIDER") or "").lower()
    if not provider:
        if os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            provider = "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "offline"
    if provider == "offline":
        return OfflineLLM()
    try:
        return _APILLM(provider, model)
    except Exception as exc:
        print(f"[generator] Could not start the {provider} client ({type(exc).__name__}: {exc}); "
              f"using offline template answers.")
        return OfflineLLM()
