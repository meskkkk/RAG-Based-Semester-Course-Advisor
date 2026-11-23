"""BM - the course-registration chatbot (task B6).

BM talks with a student who wants to register their courses.

* A NEW student is interviewed step by step (name, level, semester, passed courses,
  GPA, skills, interests/goals, desired credit load) - this is the student-profile
  input of the assignment.  The answers go through Member A's Profile.validate(),
  so missing or wrong values are flagged (R9), never guessed.
* A demo student (S01-S06, the synthetic test profiles) can be picked instead.
* Course questions are answered by Member A's rule checker, so every yes/no is
  correct and cites its rule; open questions are answered by the LLM using the
  retrieved passages.  Every factual reply ends with its sources.

What the student can say (examples):
    "recommend courses for me"          full RAG recommendation
    "can I take CS402?"                 eligibility check against the current plan
    "add CS310" / "drop CS304"          change the plan (adding is rule-checked)
    "show my plan"                      current plan + total credits
    "why was CS301 rejected?"           reasons + rule passages
    "tell me about CS305"               course passage + eligibility
    "list courses" / "level 2 courses" / "spring courses"   browse the catalog
    "what is the credit limit?"         rule passages + LLM answer
    "my gpa is 3.6" / "I passed MATH202" / "my level is 3"  update the profile
    "start over"                        restart the interview / reset the plan
    "help", "hi", "thanks", ...         small talk

Public API (shared interface contract):
    chat(message, history, profile) -> reply

Extra:
    ChatSession(profile, llm=None).ask(message) -> reply
    ChatSession.new_student(llm=None)          session that starts with the interview
    run_console_chat(profile=None)             typed chat (None = new student)
    build_gradio_app(profiles, llm=None)       BM web interface (gradio Blocks)
    launch_app(profiles, llm=None, share=False) build + launch with the BM theme
"""
from __future__ import annotations

import copy
import html
import importlib.util
import inspect
import re
import sys
from pathlib import Path

from kb_loader import load_courses
from rules import check_course, check_plan, credit_limits
from retriever import retrieve, course_evidence, rule_evidence
from generator import CHAT_SYSTEM_PROMPT, build_question_prompt, get_llm
from recommender import recommend

BOT_NAME = "BM"
_SRC = Path(__file__).resolve().parent
_CODE_RE = re.compile(r"\b([A-Za-z]{2,5})\s?-?(\d{3})\b")
_RULE_RE = re.compile(r"\bR([1-9])\b")

# keywords -> rule IDs, so rule questions always include the right passage
_RULE_WORDS = {
    "credit": ["R1", "R2", "R3"], "load": ["R1", "R2", "R3"], "overload": ["R2"], "maximum": ["R1", "R2"],
    "minimum": ["R1", "R3"], "probation": ["R3"], "gpa": ["R2", "R3"], "level": ["R4"],
    "prerequisite": ["R5"], "prereq": ["R5"], "semester": ["R6"], "offered": ["R6"], "spring": ["R6"],
    "fall": ["R6"], "clash": ["R7"], "conflict": ["R7"], "overlap": ["R7"], "time": ["R7"],
    "retake": ["R8"], "repeat": ["R8"], "again": ["R8"], "missing": ["R9"],
}

_LEVEL_WORDS = {"first": 1, "freshman": 1, "one": 1, "1st": 1, "second": 2, "sophomore": 2, "two": 2, "2nd": 2,
                "third": 3, "junior": 3, "three": 3, "3rd": 3, "fourth": 4, "senior": 4, "four": 4, "4th": 4}
_SKIP = re.compile(r"^\s*(skip|no idea|don'?t know|dont know|not sure|idk|-|n/?a|none)\s*[.!]?\s*$", re.I)
_NONE = re.compile(r"^\s*(none|no|nothing|no courses?|not yet|i'?m new|new student|0)\s*[.!]?\s*$", re.I)

HELP_TEXT = f"""Here is what I can do for you:

- **recommend courses for me**: I build a plan that fits your interests and every registration rule
- **can I take CS402?**: I check one course against the rules and your current plan
- **add CS310** / **drop CS304**: change your plan (I check the rules before adding)
- **show my plan**: your courses and total credit hours
- **why was CS301 rejected?**: the exact rule and reason
- **tell me about CS305**: what the course covers and whether you can take it
- **list courses**, **level 2 courses**, **spring courses**: browse the catalog
- **what is the credit limit?**, **what is probation?**: questions about the rules
- **my gpa is 3.6**, **I passed MATH202**: update your profile
- **start over**: begin again"""


def _profile_module():
    """Member A's src/profile.py (its name clashes with Python's built-in 'profile' module)."""
    name = "course_advisor_profile"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SRC / "profile.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cite(items) -> str:
    srcs = list(dict.fromkeys(i["source"] for i in items))
    return ("\n\nSources: " + ", ".join(srcs)) if srcs else ""


def _split_list(text: str) -> list:
    parts = re.split(r",|;|/|\band\b|\n", text)
    return [p.strip(" .!") for p in parts if p.strip(" .!")]


# ============================================================== chat session
class ChatSession:
    # interview questions for a new student, in order
    STEPS = ["name", "level", "semester", "passed", "gpa", "skills", "interests", "credits"]

    def __init__(self, profile, llm=None, interview: bool = False):
        self.profile = copy.deepcopy(profile)     # profile edits in the chat do not change the original
        self.llm = llm or get_llm()
        self.courses = load_courses()
        self.plan = []              # accepted course codes
        self.last_rec = None
        self.history = []           # [(user, bot)]
        self.step = 0 if interview else None      # index into STEPS while interviewing
        self.messages = [{"role": "assistant", "content": self.welcome()}]

    @classmethod
    def new_student(cls, llm=None) -> "ChatSession":
        pm = _profile_module()
        return cls(pm.Profile(id="NEW", passed=None), llm, interview=True)

    @property
    def interviewing(self) -> bool:
        return self.step is not None

    # ------------------------------------------------------------ greetings
    def welcome(self) -> str:
        if self.interviewing:
            return (f"Hi, I'm **{BOT_NAME}**, your course-registration assistant. "
                    "I'll ask you a few quick questions about your studies, then suggest courses that fit "
                    "your goals and follow every registration rule.\n\n"
                    "First, **what's your name?**")
        name = (self.profile.name or "there").split()[0]
        return (f"Hi {name}, I'm **{BOT_NAME}**, your course-registration assistant. "
                f"I have your profile (level {self.profile.level}, {self.profile.semester} semester). "
                "Say **recommend courses for me** to get a plan, or type **help** to see everything I can do.")

    # ------------------------------------------------------------ helpers
    def _codes(self, text):
        return [f"{a.upper()}{b}" for a, b in _CODE_RE.findall(text)]

    def _first_name(self):
        return (self.profile.name or "").split()[0] if self.profile.name else ""

    def _plan_text(self) -> str:
        if not self.plan:
            return "Your plan is empty. Say **recommend courses for me**, or **add** a course code (for example **add CS204**)."
        lim = credit_limits(self.profile.gpa)
        total = sum(self.courses[c].credits for c in self.plan)
        rows = [f"- {c} {self.courses[c].title} ({self.courses[c].credits} cr)" for c in self.plan]
        warn = f"\n\nNote: this is below the minimum of {lim['min']} credit hours ({lim['rule']})." if total < lim["min"] else ""
        return f"Your current plan ({total} credit hours, allowed {lim['min']}-{lim['max']}):\n" + "\n".join(rows) + warn

    def _llm_answer(self, question, passages, facts=""):
        text = self.llm.generate(CHAT_SYSTEM_PROMPT, build_question_prompt(question, self.profile, passages, facts))
        if text:
            return text
        # offline: give the facts and the most relevant passages directly
        body = (facts + "\n\n" if facts else "") + "Here is what the registration documents say:\n"
        for p in passages[:3]:
            body += f"- {p['text'].splitlines()[-1][:300]} [{p['source']}]\n"
        return body.strip()

    def _catalog(self, msg: str) -> str:
        low = msg.lower()
        level = None
        m = re.search(r"level\s*([1-4])|\b([1-4])(?:st|nd|rd|th)?\s*(?:level|year)", low)
        if m:
            level = int(m.group(1) or m.group(2))
        sem = "Fall" if "fall" in low else "Spring" if "spring" in low else None
        rows, cur = [], None
        for c in sorted(self.courses.values(), key=lambda c: (c.level, c.code)):
            if (level and c.level != level) or (sem and sem not in c.offered):
                continue
            if c.level != cur:
                cur = c.level
                rows.append(f"\n**Level {c.level}**")
            pre = ", ".join(c.prerequisites) or "none"
            rows.append(f"- **{c.code}** {c.title}: {c.credits} cr, {'/'.join(c.offered)}, prerequisites: {pre}")
        title = "Course catalog" + (f", level {level}" if level else "") + (f", {sem}" if sem else "")
        if not rows:
            return f"No courses match ({title})."
        return f"**{title}** ({sum(1 for r in rows if r.startswith('- '))} courses):" + "\n".join(rows) + \
               "\n\nSources: courses.json, course_descriptions.md"

    # ------------------------------------------------------------ interview
    def _question(self, step: str) -> str:
        name = self._first_name()
        return {
            "name": "What's your name?",
            "level": f"Nice to meet you{', ' + name if name else ''}! **Which academic level are you in?** (1, 2, 3 or 4)",
            "semester": "**Which semester are you registering for?** Fall or Spring?",
            "passed": ("**Which courses have you already passed?** Type the course codes separated by commas "
                       "(for example: CS101, MATH101, ENG101), or **none** if this is your first semester. "
                       "Type **list** to see the course catalog."),
            "gpa": ("**What is your cumulative GPA** (0.0 to 4.0)? It decides your credit limit. "
                    "Type **skip** if you don't know it; then only the standard limits apply."),
            "skills": "**What are your skills or strengths?** (for example: python, math, design), or **skip**",
            "interests": "**What are your interests or career goals?** (for example: AI, security, web development, data)",
            "credits": None,     # built below because it depends on the GPA
        }[step]

    def _credits_question(self) -> str:
        lim = credit_limits(self.profile.gpa)
        return (f"Last question: **how many credit hours would you like this semester?** "
                f"With your GPA the allowed range is **{lim['min']}-{lim['max']}** ({lim['rule']}).")

    def _ask_step(self) -> str:
        step = self.STEPS[self.step]
        return self._credits_question() if step == "credits" else self._question(step)

    def _interview(self, msg: str) -> str:
        step = self.STEPS[self.step]
        text = msg.strip()
        low = text.lower()
        p = self.profile
        note = ""

        if step == "name":
            name = re.sub(r"^(my name is|i am|i'm|im|it's|its|call me)\s+", "", text, flags=re.I).strip(" .!")
            p.name = name.title()[:40] if name and not _SKIP.match(name) else "Student"

        elif step == "level":
            m = re.search(r"\b([1-4])\b", low)
            lvl = int(m.group(1)) if m else next((v for k, v in _LEVEL_WORDS.items() if re.search(rf"\b{k}\b", low)), None)
            if lvl is None:
                return "Please give your level as a number from **1** to **4**."
            p.level = lvl

        elif step == "semester":
            if re.search(r"fall|autumn", low):
                p.semester = "Fall"
            elif "spring" in low:
                p.semester = "Spring"
            else:
                return "Please answer **Fall** or **Spring**."

        elif step == "passed":
            if re.fullmatch(r"\s*(list|catalog|show (me )?(the )?courses)\s*", low):
                return self._catalog("") + "\n\n" + self._question("passed")
            codes = self._codes(text)
            if not codes:
                if _NONE.match(text):
                    p.passed = []
                else:
                    return ("I couldn't find any course codes there. Please type codes like **CS101, MATH101**, "
                            "or **none** if you haven't passed any course yet.")
            else:
                known = [c for c in dict.fromkeys(codes) if c in self.courses]
                unknown = [c for c in dict.fromkeys(codes) if c not in self.courses]
                p.passed = known
                if unknown:
                    one = len(unknown) == 1
                    note = (f"Note: {', '.join(unknown)} {'is' if one else 'are'} not in the catalog, "
                            f"so I left {'it' if one else 'them'} out (R9).\n\n")
                note += f"Got it: {len(known)} passed course(s) ({', '.join(known) or 'none'}).\n\n"

        elif step == "gpa":
            if _SKIP.match(text):
                p.gpa = None
                note = "No problem, I'll use the standard credit limits.\n\n"
            else:
                m = re.search(r"\d+(?:\.\d+)?", text)
                if not m or not (0 <= float(m.group()) <= 4):
                    return "Please type a GPA between **0.0** and **4.0**, or **skip**."
                p.gpa = float(m.group())

        elif step in ("skills", "interests"):
            items = [] if _SKIP.match(text) else _split_list(text)
            setattr(p, step, items)

        elif step == "credits":
            m = re.search(r"\d+", text)
            if not m or not (1 <= int(m.group()) <= 30):
                return "Please type a number of credit hours, for example **15**."
            p.desired_credits = int(m.group())
            lim = credit_limits(p.gpa)
            if not lim["min"] <= p.desired_credits <= lim["max"]:
                note = (f"Heads-up: {p.desired_credits} is outside your allowed range of {lim['min']}-{lim['max']} "
                        f"({lim['rule']}), so I'll keep your plan inside that range.\n\n")

        # next question, or finish
        self.step += 1
        if self.step < len(self.STEPS):
            return note + self._ask_step()
        self.step = None
        p.validate(self.courses)
        flags = [f for f in p.flags if "desired_credits" not in f]
        summary = (f"{note}Thanks{', ' + self._first_name() if self._first_name() else ''}! Here is your profile:\n"
                   f"- Level {p.level}, {p.semester} semester, GPA {p.gpa if p.gpa is not None else 'not given'}\n"
                   f"- Passed: {', '.join(p.passed) if p.passed else 'no courses yet'}\n"
                   f"- Skills: {', '.join(p.skills) or '-'}; interests: {', '.join(p.interests) or '-'}\n"
                   f"- Wants {p.desired_credits} credit hours\n")
        if flags:
            summary += "\nPlease note: " + " | ".join(flags) + "\n"
        return summary + "\nLet me find the best courses for you...\n\n" + self._recommend()

    # ------------------------------------------------------------ intents
    def _recommend(self):
        rec = recommend(self.profile, llm=self.llm)
        self.last_rec = rec
        self.plan = [c["code"] for c in rec["courses"]]
        lines = [rec["summary"], "", self._plan_text()]
        if rec["flags"]:
            lines.append("\nFlags: " + " | ".join(rec["flags"]))
        lines.append("\nYou can ask **why was CS301 rejected?**, **add CS310** or **drop CS304** to adjust the plan.")
        return "\n".join(lines) + "\n\nSources: " + ", ".join(rec["sources"])

    def _check(self, code, add=False):
        if code not in self.courses:
            ev = rule_evidence(["R9"])
            return f"{code} is not in the course catalog, so I cannot check it (R9). Please check the code." + _cite(ev)
        if code in self.plan:
            return f"{code} is already in your plan.\n\n" + self._plan_text()
        res = check_plan(self.profile, self.plan + [code])
        rej = [r for r in res["rejected"] if r["code"] == code]
        c = self.courses[code]
        ev = course_evidence(code)
        if rej:
            r = rej[0]
            ev += rule_evidence(r["rule_refs"])
            reasons = "\n".join(f"- {x}" for x in r["reasons"])
            tip = f"\n\nYou need to pass {', '.join(r['missing'])} first." if r["missing"] else ""
            return f"No, you cannot {'add' if add else 'take'} {code} {c.title} right now:\n{reasons}{tip}" + _cite(ev)
        ev += rule_evidence(["R4", "R5", "R6", "R7", credit_limits(self.profile.gpa)["rule"]])
        if add:
            self.plan.append(code)
            return f"Done. {code} {c.title} passed every rule check and was added.\n\n{self._plan_text()}" + _cite(ev)
        return (f"Yes, {code} {c.title} ({c.credits} cr) is allowed with your current plan: level, prerequisites, "
                f"semester, time slots and credit limit are all fine. Say **add {code}** to add it." + _cite(ev))

    def _drop(self, code):
        if code in self.plan:
            self.plan.remove(code)
            return f"Removed {code}.\n\n" + self._plan_text()
        return f"{code} is not in your plan.\n\n" + self._plan_text()

    def _why_rejected(self, code):
        if self.last_rec:
            for x in self.last_rec["excluded"]:
                if x["code"] == code:
                    ev = course_evidence(code) + rule_evidence(x["rule_refs"])
                    return f"{code} was not included because:\n" + "\n".join(f"- {r}" for r in x["reasons"]) + _cite(ev)
        single = check_course(self.profile, code)
        if not single["eligible"]:
            ev = course_evidence(code) + rule_evidence(single["rule_refs"])
            return f"{code} is not allowed for you:\n" + "\n".join(f"- {r}" for r in single["reasons"]) + _cite(ev)
        return self._check(code)

    def _about(self, code):
        if code not in self.courses:
            return f"{code} is not in the course catalog." + _cite(rule_evidence(["R9"]))
        ev = course_evidence(code)
        status = check_course(self.profile, code)
        note = "You are eligible for it." if status["eligible"] else "You are not eligible yet: " + "; ".join(status["reasons"])
        return ev[0]["text"] + "\n\n" + note + _cite(ev + rule_evidence(status["rule_refs"]))

    def _update_profile(self, msg):
        low = msg.lower()
        changed = []
        m = re.search(r"gpa\s*(?:is|=|:)?\s*([0-4](?:\.\d+)?)", low)
        if m:
            self.profile.gpa = float(m.group(1)); changed.append(f"GPA = {m.group(1)}")
        m = re.search(r"level\s*(?:is|=|:)?\s*([1-4])\b", low)
        if m:
            self.profile.level = int(m.group(1)); changed.append(f"level = {m.group(1)}")
        m = re.search(r"(\d{1,2})\s*(?:credit|credits|hours)", low)
        if m and re.search(r"\b(want|desire|prefer|take)\b", low):
            self.profile.desired_credits = int(m.group(1)); changed.append(f"desired credits = {m.group(1)}")
        m = re.search(r"\b(fall|spring)\b", low)
        if m and re.search(r"semester", low):
            self.profile.semester = m.group(1).capitalize(); changed.append(f"semester = {self.profile.semester}")
        if re.search(r"\b(passed|completed|finished)\b", low):
            codes = self._codes(msg)
            if codes:
                self.profile.passed = list(dict.fromkeys((self.profile.passed or []) + codes))
                changed.append("passed += " + ", ".join(codes))
        if not changed:
            return None
        self.profile.validate(self.courses)
        self.plan = [c for c in self.plan if c in check_plan(self.profile, self.plan)["accepted"]]
        flags = ("\nStill flagged: " + " | ".join(self.profile.flags)) if self.profile.flags else ""
        return "Updated your profile: " + "; ".join(changed) + "." + flags + \
               "\n\nSay **recommend courses for me** to get a new plan."

    def _small_talk(self, low: str):
        name = self._first_name()
        if re.fullmatch(r"(hi+|hello|hey+|hiya|salam|salaam|assalamu alaikum|ahlan|good (morning|afternoon|evening))[ !.,]*(b&m|there)?[ !.]*", low):
            return (f"Hello{' ' + name if name else ''}! How can I help with your registration? "
                    "Try **recommend courses for me**, or type **help**.")
        if re.fullmatch(r"(thanks?( you)?|thank u|thx|ty|great|perfect|awesome|ok(ay)?|cool|nice)[ !.]*( b&m)?[ !.]*", low):
            return "You're welcome! Anything else about your courses?"
        if re.fullmatch(r"(bye|goodbye|see you|that'?s all|done)[ !.]*", low):
            return f"Good luck with your registration{', ' + name if name else ''}! Come back any time."
        if re.fullmatch(r"(help|what can you do\??|commands|menu|\?)", low):
            return HELP_TEXT
        if re.fullmatch(r"(who are you|what are you|what is b&m|what'?s your name)\??", low):
            return (f"I'm **{BOT_NAME}**, a course-registration assistant built by team BM. I read the course catalog "
                    "and the registration rules, check every course against the rules, and explain my suggestions "
                    "with sources.")
        return None

    def reset(self) -> str:
        self.plan, self.last_rec = [], None
        if self.profile.id == "NEW":
            pm = _profile_module()
            self.profile = pm.Profile(id="NEW", passed=None)
            self.step = 0
            return "Let's start over. " + self._question("name")
        return "Your plan is cleared. Say **recommend courses for me** to build a new one."

    # ------------------------------------------------------------ router
    def ask(self, message: str) -> str:
        msg = (message or "").strip()
        low = msg.lower()
        codes = self._codes(msg)
        reply = None
        if not msg:
            reply = "Please type a message."
        elif re.fullmatch(r"(start over|restart|reset|new profile|start again)[ !.]*", low):
            reply = self.reset()
        elif self.interviewing:
            st = self._small_talk(low)
            if st and not st.startswith("Here is what I can do"):
                st = "Hello! Happy to help." if st.startswith("Hello") else st.split("!")[0] + "!"
            reply = (st + "\n\n" + self._ask_step()) if st else self._interview(msg)
        elif (st := self._small_talk(low)) is not None:
            reply = st
        elif re.search(r"\b(my )?(gpa|level)\b.*\b(is|=)\b|\bi (have )?(passed|completed|finished)\b|\bi want\b.*\bcredit", low) \
                and (upd := self._update_profile(msg)):
            reply = upd
        elif re.search(r"\b(recommend|suggest|what should i (take|register)|plan for me|which courses (should|can) i)\b", low):
            reply = self._recommend()
        elif re.search(r"\b(list|show|browse|all|available)\b.*\bcourses\b|\bcatalog\b|"
                       r"\bcourses?\b.*\b(level [1-4]|in (fall|spring)|offered)\b|\b(level [1-4]|fall|spring) courses\b",
                       low) and not codes:
            reply = self._catalog(msg)
        elif re.search(r"\b(show|my|current)\b.*\bplan\b", low) and not codes:
            reply = self._plan_text()
        elif codes and re.search(r"\b(drop|remove|delete)\b", low):
            reply = "\n\n".join(self._drop(c) for c in codes)
        elif codes and re.search(r"\b(add|register)\b", low) and not re.search(r"\bcan i\b", low):
            reply = "\n\n".join(self._check(c, add=True) for c in codes)
        elif codes and re.search(r"\b(why|reject|excluded|not allowed|not included)\b", low):
            reply = "\n\n".join(self._why_rejected(c) for c in codes)
        elif codes and re.search(r"\b(can i|am i|eligible|allowed|take)\b", low):
            reply = "\n\n".join(self._check(c) for c in codes)
        elif codes and (re.search(r"\b(about|what is|describe|tell me|info)\b", low) or len(msg.split()) <= 2):
            reply = "\n\n".join(self._about(c) for c in codes)
        else:
            rule_ids = [f"R{n}" for n in _RULE_RE.findall(msg)]
            for word, ids in _RULE_WORDS.items():
                if word in low:
                    rule_ids += ids
            passages = rule_evidence(rule_ids) + retrieve(msg, k=4)
            seen, uniq = set(), []
            for p in passages:
                if p["source"] not in seen:
                    seen.add(p["source"]); uniq.append(p)
            facts = ""
            if set(rule_ids) & {"R1", "R2", "R3"}:
                lim = credit_limits(self.profile.gpa)
                facts = f"Your credit limit is {lim['min']}-{lim['max']} credit hours ({lim['rule']}, {lim['status']})."
            reply = self._llm_answer(msg, uniq[:6], facts) + _cite(uniq[:6])
        self.history.append((message, reply))
        self.messages += [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
        return reply


# ---------------------------------------------------------- contract function
_SESSIONS = {}


def chat(message, history=None, profile=None, llm=None) -> str:
    """Contract entry point: one ChatSession per student id (profile=None starts a new-student interview)."""
    key = profile.id if profile is not None else "__new__"
    if key not in _SESSIONS:
        _SESSIONS[key] = ChatSession(profile, llm) if profile is not None else ChatSession.new_student(llm)
    return _SESSIONS[key].ask(message)


def run_console_chat(profile=None, llm=None):
    """Typed chat in a terminal or notebook; profile=None starts as a new student. Type 'quit' to stop."""
    s = ChatSession(profile, llm) if profile is not None else ChatSession.new_student(llm)
    print(f"{BOT_NAME}:", s.messages[0]["content"])
    while True:
        msg = input("You: ")
        if msg.strip().lower() in ("quit", "exit"):
            print(f"{BOT_NAME}: Good luck with your registration!")
            return s
        print(f"{BOT_NAME}:", s.ask(msg))


def load_demo_profiles(directory=None) -> dict:
    """{'S01 - Nour Hassan': Profile, ...} from the profiles/ folder."""
    directory = Path(directory) if directory else _SRC.parent / "profiles"
    pm = _profile_module()
    return {f"{p.id} - {p.name}": p for p in pm.load_profiles(directory)}


# ================================================================== web UI
NEW_STUDENT = "New student (enter my own details)"

CSS = """
/* ---------- dark page with soft colour glows (so the glass has something to blur) ---------- */
body, .gradio-container, .dark {
  --body-background-fill: #070b18; --background-fill-primary: rgba(15,23,42,.55);
  --background-fill-secondary: rgba(30,41,59,.45); --block-background-fill: rgba(15,23,42,.55);
  --border-color-primary: rgba(148,163,184,.16); --block-border-color: rgba(148,163,184,.16);
  --input-background-fill: rgba(15,23,42,.75); --body-text-color: #e2e8f0; --body-text-color-subdued: #94a3b8;
  --block-label-background-fill: rgba(99,102,241,.18); --block-label-text-color: #c7d2fe;
  --block-info-text-color: #94a3b8; --button-secondary-background-fill: rgba(30,41,59,.7);
  --button-secondary-background-fill-hover: rgba(51,65,85,.9); --button-secondary-text-color: #e2e8f0;
  --button-secondary-border-color: rgba(148,163,184,.2);
  --color-accent-soft: rgba(99,102,241,.18); --chatbot-text-size: 15px;
}
body, gradio-app {
  background: radial-gradient(900px 600px at 8% 0%, rgba(79,70,229,.35), transparent 60%),
              radial-gradient(800px 600px at 100% 20%, rgba(13,148,136,.28), transparent 60%),
              radial-gradient(700px 500px at 50% 110%, rgba(147,51,234,.22), transparent 60%),
              #070b18 !important;
  background-attachment: fixed !important;
}
.gradio-container {max-width: 1320px !important; margin: auto; padding-top: 96px !important; background: transparent !important;}
footer {opacity: .5;}

/* ---------- full-width glass navbar ---------- */
#bm-nav {position: fixed; top: 0; left: 0; right: 0; z-index: 1000; height: 78px;
  display: flex; align-items: center; justify-content: space-between; padding: 0 clamp(16px, 4vw, 56px);
  background: linear-gradient(90deg, rgba(30,27,75,.55), rgba(15,23,42,.45) 50%, rgba(19,78,74,.45));
  backdrop-filter: blur(18px) saturate(170%); -webkit-backdrop-filter: blur(18px) saturate(170%);
  border-bottom: 1px solid rgba(255,255,255,.10); box-shadow: 0 10px 40px rgba(0,0,0,.35);}
#bm-nav .brand {display: flex; align-items: center; gap: 14px;}
#bm-logo {width: 44px; height: 44px; flex-shrink: 0; display: flex; align-items: center; justify-content: center;}
#bm-logo svg {display: block;}
#bm-nav h1 {margin: 0; font-size: 20px; font-weight: 700; color: #f8fafc; line-height: 1.2;}
#bm-nav p {margin: 2px 0 0; font-size: 12.5px; color: #a5b4fc;}
#bm-nav .pills {display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end;}
#bm-nav .pill {font-size: 12px; padding: 5px 12px; border-radius: 999px; color: #e2e8f0;
  background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12); white-space: nowrap;}
#bm-nav .pill.live::before {content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: #34d399; margin-right: 7px; box-shadow: 0 0 8px #34d399;}
@media (max-width: 760px) {#bm-nav .pills {display: none;} #bm-nav p {display: none;}}

/* ---------- glass panels ---------- */
.block:not(#bm-nav-wrap):not(#bm-who), .form {background: rgba(15,23,42,.50) !important; border-color: rgba(148,163,184,.14) !important;}
.block:not(#bm-nav-wrap):not(#bm-who) {backdrop-filter: blur(12px);}
/* no blur/clipping around the picker: its option list is positioned on the screen */
.form:has(#bm-who) {backdrop-filter: none !important; overflow: visible !important;}
/* the student picker must open above the glass cards below it */
#bm-who {position: relative; z-index: 60 !important; overflow: visible !important; backdrop-filter: none !important;}
#bm-who ul.options, #bm-who .options {z-index: 70 !important; background: #0f172a !important;
  border: 1px solid rgba(148,163,184,.25) !important; box-shadow: 0 16px 40px rgba(0,0,0,.55) !important;}
#bm-who li.item:hover, #bm-who li.item.active {background: rgba(99,102,241,.25) !important;}
#bm-nav-wrap {background: transparent !important; border: none !important; padding: 0 !important; margin: 0 !important;
  min-height: 0 !important; height: 0 !important; backdrop-filter: none !important; box-shadow: none !important; overflow: visible !important;}
.bm-card {border: 1px solid rgba(148,163,184,.16); border-radius: 16px; padding: 14px 16px; color: #e2e8f0;
  background: linear-gradient(160deg, rgba(30,41,59,.65), rgba(15,23,42,.55)); backdrop-filter: blur(12px);
  box-shadow: 0 8px 30px rgba(0,0,0,.25);}
.bm-card h3 {margin: 0 0 8px 0; font-size: 15px; color: #f8fafc;}
.bm-row {display: flex; justify-content: space-between; font-size: 13.5px; padding: 3px 0;}
.bm-row b {color: #f8fafc;}
.bm-muted {color: #94a3b8; font-size: 12.5px;}
.bm-chip {display: inline-block; padding: 2px 9px; margin: 2px 3px 2px 0; border-radius: 999px; font-size: 12px;
  background: rgba(99,102,241,.18); color: #c7d2fe; border: 1px solid rgba(99,102,241,.25);}
.bm-chip.teal {background: rgba(20,184,166,.15); color: #99f6e4; border-color: rgba(20,184,166,.25);}
.bm-flag {font-size: 12px; color: #fcd34d; background: rgba(245,158,11,.12); border: 1px solid rgba(245,158,11,.25);
  border-radius: 10px; padding: 6px 8px; margin-top: 8px;}
.bm-bar {height: 10px; border-radius: 99px; background: rgba(148,163,184,.18); margin: 10px 0 4px;}
.bm-bar > div {height: 100%; border-radius: 99px; background: linear-gradient(90deg, #6366f1, #14b8a6);
  box-shadow: 0 0 12px rgba(99,102,241,.6);}
.bm-course {display: flex; justify-content: space-between; font-size: 13.5px; padding: 6px 0;
  border-bottom: 1px dashed rgba(148,163,184,.18);}
.bm-course b {color: #a5b4fc;}
.bm-steps {display: flex; gap: 4px; margin-top: 8px;} .bm-steps span {flex: 1; height: 6px; border-radius: 99px;
  background: rgba(148,163,184,.2);} .bm-steps span.done {background: linear-gradient(90deg, #6366f1, #14b8a6);}

/* ---------- chat ---------- */
#bm-chat .message.bot, #bm-chat .bot {background: rgba(30,41,59,.75) !important; border: 1px solid rgba(148,163,184,.14) !important;}
#bm-chat .message.user, #bm-chat .user {background: linear-gradient(135deg, rgba(79,70,229,.85), rgba(13,148,136,.75)) !important;
  border: none !important; color: #fff !important;}
#bm-chat code {background: rgba(99,102,241,.15) !important; color: #c7d2fe !important; border: none !important;}
#bm-input textarea {font-size: 15px !important;}
#bm-quick button {font-size: 13px !important; border-radius: 999px !important;}
#bm-send {background: linear-gradient(135deg, #6366f1, #14b8a6) !important; border: none !important; color: #fff !important;}
#bm-footer {text-align: center; color: #64748b; font-size: 12.5px; margin-top: 6px;}
"""


# flat monogram logo
LOGO_SVG = """<svg width="44" height="44" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg" aria-label="BM">
<rect x="0.5" y="0.5" width="43" height="43" rx="11" fill="#111827" stroke="#334155"/>
<text x="22" y="27.5" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="16" font-weight="700"
 letter-spacing="0.5" fill="#f8fafc">BM</text>
<rect x="14" y="32" width="16" height="2" rx="1" fill="#818cf8"/></svg>"""

# always open in dark mode
DARK_JS = """
() => {
    document.body.classList.add('dark');
    document.querySelector('gradio-app')?.classList.add('dark');
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.history.replaceState(null, '', url.href);
    }
    // always open at the top of the page
    if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
    const top = () => window.scrollTo({top: 0, left: 0, behavior: 'instant'});
    top();
    [100, 400, 1000, 2000].forEach(ms => setTimeout(top, ms));
}
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _header_html(llm_name: str) -> str:
    model = llm_name.split(" (")[0]
    return f"""<nav id="bm-nav"><div class="brand"><div id="bm-logo">{LOGO_SVG}</div><div>
<h1>BM Course Advisor</h1><p>Your semester, planned with the registration rules and sources</p></div></div>
<div class="pills"><span class="pill">RAG retrieval</span><span class="pill">Rule-checked</span>
<span class="pill">Cited sources</span><span class="pill live">{_esc(model)}</span></div></nav>"""


def _profile_card(s: ChatSession) -> str:
    p = s.profile
    if s.interviewing:
        done = s.step
        bars = "".join(f'<span class="{"done" if i < done else ""}"></span>' for i in range(len(s.STEPS)))
        return (f'<div class="bm-card"><h3>Your profile</h3><div class="bm-muted">Answer BM\'s questions in the chat '
                f'({done} of {len(s.STEPS)} done).</div><div class="bm-steps">{bars}</div></div>')
    lim = credit_limits(p.gpa)
    chips = lambda xs, cls="": "".join(f'<span class="bm-chip {cls}">{_esc(x)}</span>' for x in xs) or '<span class="bm-muted">-</span>'
    passed = p.passed or []
    flags = [f for f in (p.flags or [])]
    flag_html = "".join(f'<div class="bm-flag">{_esc(f)}</div>' for f in flags[:3])
    return f"""<div class="bm-card"><h3>{_esc(p.name or 'Student')}</h3>
<div class="bm-row"><span>Level</span><b>{_esc(p.level if p.level is not None else 'missing')}</b></div>
<div class="bm-row"><span>Semester</span><b>{_esc(p.semester or 'missing')}</b></div>
<div class="bm-row"><span>GPA</span><b>{_esc(p.gpa if p.gpa is not None else 'not given')}</b></div>
<div class="bm-row"><span>Credit limit</span><b>{lim['min']}-{lim['max']} ({lim['rule']})</b></div>
<div class="bm-row"><span>Wants</span><b>{_esc(p.desired_credits or '-')} credit hours</b></div>
<div style="margin-top:8px" class="bm-muted">Passed ({len(passed)})</div><div>{chips(passed)}</div>
<div style="margin-top:6px" class="bm-muted">Interests</div><div>{chips(p.interests, 'teal')}</div>
<div style="margin-top:6px" class="bm-muted">Skills</div><div>{chips(p.skills, 'teal')}</div>{flag_html}</div>"""


def _plan_card(s: ChatSession) -> str:
    if s.interviewing:
        return '<div class="bm-card"><h3>My plan</h3><div class="bm-muted">Your plan appears here after the interview.</div></div>'
    lim = credit_limits(s.profile.gpa)
    total = sum(s.courses[c].credits for c in s.plan)
    pct = min(100, round(100 * total / lim["max"])) if lim["max"] else 0
    rows = "".join(f'<div class="bm-course"><span><b>{c}</b> {_esc(s.courses[c].title)}</span><span>{s.courses[c].credits} cr</span></div>'
                   for c in s.plan) or '<div class="bm-muted">Empty: click <b>Recommend courses</b>.</div>'
    status = "below minimum" if total < lim["min"] else "within limits"
    return f"""<div class="bm-card"><h3>My plan</h3>{rows}
<div class="bm-bar"><div style="width:{pct}%"></div></div>
<div class="bm-row"><span>{total} credit hours</span><span class="bm-muted">allowed {lim['min']}-{lim['max']}, {status}</span></div></div>"""


def _pretty(text: str) -> str:
    """Show the sources line smaller in the chat window."""
    def fmt(m):
        items = [x.strip() for x in m.group(1).split(",") if x.strip()]
        return "\n\n**Sources:** " + " ".join(f"`{x}`" for x in items)
    return re.sub(r"\n\nSources: (.+)$", fmt, text, flags=re.S)


def build_gradio_app(profiles: dict, llm=None):
    """BM web interface. ``profiles`` = {label: Profile}. Use launch_app() to start it with the theme."""
    import gradio as gr

    llm = llm or get_llm()
    choices = [NEW_STUDENT] + list(profiles)

    def session_for(sessions, who):
        if who not in sessions:
            sessions[who] = ChatSession.new_student(llm) if who == NEW_STUDENT else ChatSession(profiles[who], llm)
        return sessions[who]

    def view(sessions, who):
        s = session_for(sessions, who)
        msgs = [{"role": m["role"], "content": _pretty(m["content"]) if m["role"] == "assistant" else m["content"]}
                for m in s.messages]
        return sessions, msgs, _profile_card(s), _plan_card(s)

    def send(text, sessions, who):
        if text and text.strip():
            session_for(sessions, who).ask(text)
        return ("",) + view(sessions, who)

    def quick(label):
        return lambda sessions, who: send(label, sessions, who)

    chatbot_kwargs = dict(height="calc(100vh - 330px)", min_height=320, elem_id="bm-chat", label=f"Chat with {BOT_NAME}", show_label=False,
                          placeholder=f"<b>{BOT_NAME}</b><br>Your course-registration assistant")
    if "type" in inspect.signature(gr.Chatbot.__init__).parameters:   # gradio 5
        chatbot_kwargs["type"] = "messages"
    blocks_kwargs = {"title": "BM Course Advisor"}
    if "css" in inspect.signature(gr.Blocks.__init__).parameters:     # gradio 5: style goes to Blocks
        blocks_kwargs.update(css=CSS, theme=_theme(gr), js=DARK_JS)

    with gr.Blocks(**blocks_kwargs) as app:
        sessions = gr.State({})
        gr.HTML(_header_html(llm.name), elem_id="bm-nav-wrap")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                who = gr.Dropdown(choices, value=NEW_STUDENT, label="Who is registering?",
                                  info="Start as a new student, or open one of the six demo profiles.",
                                  elem_id="bm-who")
                profile_html = gr.HTML()
                plan_html = gr.HTML()
            with gr.Column(scale=2, min_width=420):
                chat_ui = gr.Chatbot(**chatbot_kwargs)
                with gr.Row():
                    box = gr.Textbox(placeholder="Type your message... (try: help)", show_label=False,
                                     scale=8, autofocus=False, container=False, elem_id="bm-input")
                    send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80, elem_id="bm-send")
                with gr.Row(elem_id="bm-quick"):
                    buttons = [(gr.Button(lbl, size="sm"), msg) for lbl, msg in [
                        ("Recommend courses", "recommend courses for me"),
                        ("My plan", "show my plan"),
                        ("Credit rules", "what is my credit limit?"),
                        ("Course catalog", "list courses"),
                        ("Help", "help"),
                        ("Start over", "start over")]]
        gr.HTML(f'<div id="bm-footer">BM Course Advisor | RAG over the course catalog and registration rules '
                f'(fictional sample data) | LLM: {_esc(llm.name)}</div>')

        outs = [sessions, chat_ui, profile_html, plan_html]
        app.load(view, [sessions, who], outs)
        who.change(view, [sessions, who], outs)
        box.submit(send, [box, sessions, who], [box] + outs)
        send_btn.click(send, [box, sessions, who], [box] + outs)
        for btn, msg in buttons:
            btn.click(quick(msg), [sessions, who], [box] + outs)
    return app


def _theme(gr):
    return gr.themes.Soft(primary_hue="indigo", secondary_hue="teal", neutral_hue="slate",
                          font=[gr.themes.GoogleFont("Inter"), gr.themes.Font("ui-sans-serif"),
                                gr.themes.Font("system-ui"), gr.themes.Font("sans-serif")])


def launch_app(profiles: dict | None = None, llm=None, share: bool = False, **kwargs):
    """Build the BM interface and open it (works with gradio 5 and 6)."""
    import gradio as gr

    app = build_gradio_app(profiles if profiles is not None else load_demo_profiles(), llm)
    if "css" in inspect.signature(gr.Blocks.launch).parameters:      # gradio 6: style goes to launch()
        kwargs.setdefault("css", CSS)
        kwargs.setdefault("theme", _theme(gr))
        kwargs.setdefault("js", DARK_JS)
    return app.launch(share=share, **kwargs)


if __name__ == "__main__":
    launch_app(inbrowser=True)
