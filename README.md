# BM: RAG-Based Semester Course Advisor

**BM** recommends which courses a student can register for next semester, and explains every decision with sources. A student can also chat with BM: a new student enters their own profile in the chat, gets a plan, and adjusts it.

> **All course data and registration rules are FICTIONAL sample data** created for this assignment. All student profiles are synthetic.

**Contents:** [Quick start](#quick-start) · [How it works](#how-it-works) · [Project layout](#project-layout) · [Setup](#setup) · [LLM configuration](#llm-configuration) · [Running the system](#running-the-system) · [Chatting with BM](#chatting-with-bm) · [Registration rules](#registration-rules) · [Demo students](#demo-students) · [Tests](#tests) · [Interface contract](#interface-contract) · [Troubleshooting](#troubleshooting) · [Limitations](#limitations) · [Team](#team)

---

## Quick start

```bash
cd course_advisor
pip install -r requirements.txt
jupyter notebook course_advisor.ipynb      # Kernel -> Restart & Run All
```

That is enough to run the whole project. No API key is needed: without one, every decision, reason and source is the same and only the advisor's message is a template instead of LLM text (see [LLM configuration](#llm-configuration)).

To check the rule checker alone, with nothing installed except Python:

```bash
python tests/test_rules.py
```

---

## How it works

```
 student profile --> profile query --> RETRIEVE -----------> course ranking + evidence passages
 (level, GPA,        (interests,        FAISS cosine search          |
  passed courses,     skills, level,    over the knowledge base      v
  interests,          semester)                                RULE CHECK (rules.py, R1-R9)
  desired credits)                                             accepted / rejected / flags
                                                                     |
                                       rule passages by rule ID -----+
                                                                     v
                                                              GENERATE (LLM): explanation
                                                              with [source] citations
                                                                     |
                                                                     v
                                                       recommendation output + BM chatbot
```

| Step | What happens |
|---|---|
| **Retrieve** | The knowledge base (courses, descriptions, rules) is split into chunks, embedded, and stored in a FAISS index. A query built from the student's interests, skills, level and semester finds the most relevant courses. |
| **Rule check** | Every candidate plan goes through `check_plan()`. Prerequisites, credit limits, level, semester availability, time clashes and repeated courses are checked by explicit Python code. Nothing reaches the student without passing it. |
| **Generate** | The LLM receives the profile, the rule-checker result (marked as final) and the evidence passages, and writes a short explanation that cites its sources. |
| **Output** | Accepted courses (with reason, eligibility note and sources), excluded courses with reasons and rule IDs, total credit hours, flags, and the retrieved passages. |

**Design decision: the LLM never decides.** Eligibility depends on exact facts (prerequisites, GPA thresholds, time overlaps) that a language model can misread. Deterministic code makes the decision and the model only explains it, so the structured output stays correct even if the model's wording is wrong, and the system still runs with no model at all.

**Missing information is flagged, not guessed.** If the level, semester or passed-course list is missing, or a course code is unknown, the system raises a flag (rule R9) and does not approve any course that depends on it.

**Two recommendation modes**
- *Requested*: the student names the courses they want; the plan is checked as given.
- *Auto*: BM builds the plan from the most relevant courses, keeping each one only if `check_plan()` still accepts the plan, until the desired credit load is reached.

---

## Project layout

```
course_advisor/
├── course_advisor.ipynb      MAIN NOTEBOOK: run all cells top to bottom (sections 0-9)
├── requirements.txt          Python dependencies
├── README.md                 this file
│
├── knowledge_base/           courses.json, course_descriptions.md, registration_rules.md
├── profiles/                 student_01.json ... student_06.json, expected_results.md / .json
│
├── src/
│   ├── kb_loader.py          load and validate the knowledge base
│   ├── profile.py            student profile, validation, interactive input
│   ├── rules.py              rule checker, rules R1-R9
│   ├── chunker.py            parse and chunk the knowledge base
│   ├── vector_store.py       embeddings and FAISS index
│   ├── retriever.py          retrieval and course ranking
│   ├── generator.py          LLM providers and prompts
│   ├── recommender.py        recommend(): retrieve -> rule check -> generate -> output
│   └── chatbot.py            BM chatbot: new-student interview, chat, web interface
│
├── tests/                    test_rules.py (32 tests), test_rag.py (10 tests)
├── outputs/                  sample outputs for the six demo profiles (created by the notebook)
└── index/                    saved vector index (created automatically)
```

| Part | Files | Built by |
|---|---|---|
| Knowledge base, student profiles, rule checking | `knowledge_base/`, `profiles/`, `kb_loader.py`, `profile.py`, `rules.py`, `tests/test_rules.py` | Mesk Khaled |
| RAG pipeline, recommendation output, chatbot | `chunker.py`, `vector_store.py`, `retriever.py`, `generator.py`, `recommender.py`, `chatbot.py`, `tests/test_rag.py` | Basmala Moataz |

---

## Setup

**Requirements:** Python 3.10 or newer. The rule-checking part (`kb_loader.py`, `profile.py`, `rules.py`, `tests/test_rules.py`) uses only the standard library. The RAG pipeline, LLM providers and chat interface need the packages in `requirements.txt`.

**Local install** (a virtual environment is recommended):

```bash
cd course_advisor
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Google Colab:**

```python
# upload the folder (or the zip and unzip it), then:
%cd course_advisor
!pip install -r requirements.txt
```

Then open `course_advisor.ipynb` and run the cells.

---

## LLM configuration

The chatbot and the advisor messages use an LLM. Get a free key from [Groq](https://console.groq.com/keys) or [Google AI Studio](https://aistudio.google.com/apikey) and set it **before section 6 of the notebook**:

```python
import os
os.environ["GROQ_API_KEY"] = "gsk_..."        # or os.environ["GEMINI_API_KEY"] = "..."
```

| Variable | Meaning |
|---|---|
| `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY` | The provider is chosen from the key you set. |
| `LLM_PROVIDER` | Force `groq`, `gemini`, `openai` or `offline`. |
| `LLM_MODEL` | Override the model. Defaults: `openai/gpt-oss-20b` on Groq, `gemini-2.5-flash`, `gpt-4o-mini`. Backup models are tried automatically if one is retired. |
| `ADVISOR_EMBEDDER` | `minilm` (default) or `tfidf` (offline fallback). |

**Without any key** the system still runs end to end: every decision, reason and source is the same, and the advisor message is a template built from the evidence instead of LLM text.

**Embeddings:** the default is the `all-MiniLM-L6-v2` sentence-transformers model. If it cannot be downloaded, TF-IDF embeddings are used automatically. They work offline but match only shared words, so retrieval is weaker for questions phrased very differently from the documents.

Never commit an API key. Set it in the notebook session or as an environment variable.

---

## Running the system

### 1. The notebook (main way)

`course_advisor.ipynb`: **Kernel → Restart & Run All**. It walks through the knowledge base, the six demo profiles (their outputs are saved to `outputs/`) and the chatbot (section 9.4).

### 2. The chat interface

- **In the notebook:** section 9.4. Set `LAUNCH_UI = True` and open the printed link (http://127.0.0.1:7860).
- **From a terminal:** `cd src && python chatbot.py` (opens the browser).

Choose **New student** and BM asks for your name, level, semester, passed courses, GPA, skills, interests and desired credit hours, then recommends courses. Or open one of the six demo students. The left panel shows the profile and the current plan with a credit-hour bar, and the buttons under the chat are quick actions.

### 3. From Python

```python
import sys; sys.path.insert(0, "src")
from chatbot import load_demo_profiles, ChatSession
from recommender import recommend, format_recommendation

profiles = {p.id: p for p in load_demo_profiles().values()}

# the advisor builds the plan
print(format_recommendation(recommend(profiles["S02"])))

# check a plan the student wishes for
print(format_recommendation(recommend(profiles["S02"], requested=["CS301", "CS304"])))

# ask a question
print(ChatSession(profiles["S02"]).ask("can I take CS301?"))

# interview a new student
new = ChatSession.new_student()
print(new.ask("Sara"))
```

To use only the rule checker:

```python
sys.path.insert(0, "src")
from profile import load_profile
from rules import check_plan, format_plan_result

student = load_profile("profiles/student_02.json")
print(format_plan_result(check_plan(student, ["CS301", "CS304", "CS302", "CS205", "CS307", "CS402"])))
```

---

## Chatting with BM

BM answers course questions with the rule checker, so those answers are exact, and answers open questions and small talk with the LLM over retrieved passages. Every factual reply lists its sources.

| You want to | Example messages |
|---|---|
| Get a plan | `recommend courses for me` |
| Check a course | `can I take CS305?` · `why was CS301 rejected?` |
| Change the plan | `add CS310` · `drop CS304` |
| Browse the catalog | `list courses` · `level 2 courses` · `tell me about CS401` |
| Ask about policy | `what is probation?` |
| Update your profile | `my gpa is 3.6` · `I passed MATH202` |
| Control the chat | `start over` · `help` |

Commands and profile updates are recognised from simple sentences. Unusual wording falls through to a general LLM answer.

---

## Registration rules

The rules are in `knowledge_base/registration_rules.md` (**fictional policy**). Each rejection returned by the checker carries a reason and the rule ID.

| ID | Rule |
|---|---|
| R1 | A student registers for 12-18 credit hours per semester. |
| R2 | GPA of 3.50 or more allows up to 21 credit hours. |
| R3 | GPA below 2.00 is academic probation: 9-12 credit hours. |
| R4 | Only courses at the student's level or lower. |
| R5 | All prerequisites must be passed first (no concurrent registration). |
| R6 | A course can be registered only in a semester in which it is offered. |
| R7 | No two courses may have overlapping meeting times. Back-to-back classes are allowed. |
| R8 | A passed course cannot be taken again. |
| R9 | Missing or unknown information is flagged, never guessed. |

Example result for student S02 (level 3, GPA 2.9, Fall):

```
Credit limits: 12-18 (R1, status: standard)
ACCEPTED (13 credit hours):
  + CS304  Image Processing and Computer Vision (3 cr)
  + CS302  Software Engineering (3 cr)
  + CS205  Operating Systems (3 cr)
  + CS307  Embedded Systems (4 cr)
REJECTED:
  - CS301  Machine Learning  [R5]
        R5: prerequisite(s) not passed: MATH202 (required for CS301)
  - CS402  Cybersecurity  [R4, R5]
        R4: CS402 is a level 4 course but the student is in level 3
        R5: prerequisite(s) not passed: CS303 (required for CS402)
FLAGS:
  (none)
```

---

## Demo students

Six synthetic students in `profiles/`, each with a hand-derived expected result in `profiles/expected_results.md` (and `.json`). The results below are for the *requested* mode: the plan listed in `expected_results.md` is checked as given.

| ID | Scenario | Expected outcome |
|---|---|---|
| S01 | Strong student: level 3, GPA 3.82, Fall | All 4 requested courses accepted, 13 credit hours. |
| S02 | **Unmet prerequisite**: MATH202 never passed | CS301 rejected (R5), CS402 rejected (R4, R5); 4 courses accepted, 13 credit hours. |
| S03 | Probation: GPA 1.8 | CS307 rejected (R3, would reach 13 of 12), CS304 rejected (R7, same slot as CS301); 9 credit hours. Desired load 15 is flagged. |
| S04 | **Credit limit exceeded**: level 4, GPA 3.2, wants 21 | First 6 courses fill exactly 18 credit hours; CS204 rejected (R1). Desired load 21 is flagged. |
| S05 | Time clash: level 4, GPA 3.6, Spring | CS306 rejected (R7, partial overlap with CS305 on Tuesday); 6 courses accepted, 17 credit hours. |
| S06 | Missing data: no level, no desired credits, unknown code CS999 | Nothing is approved; three R9 flags and a below-minimum flag. |

---

## Tests

```bash
python tests/test_rules.py     # 32 rule-checker tests (standard library only)
python tests/test_rag.py       # 10 RAG and chatbot tests (offline)
pytest tests/                  # all 42
```

- `test_rules.py` has a pass case and a fail case for each of rules R1-R9, validation tests for the knowledge base and the profiles, and a comparison of the six demo profiles with `expected_results.json`.
- `test_rag.py` checks that every course and rule is chunked, that retrieval returns the right passage for known questions, that the six requested plans give exactly the expected results, that every recommendation has a reason and sources, that auto plans stay within the credit limit, the main chatbot commands, and the new-student interview.

---

## Interface contract

**Rule checker and data (Mesk Khaled)**

```python
load_courses()                    -> dict[code, Course]
load_profile(path)                -> Profile
check_course(profile, code)       -> {"eligible", "reasons", "rule_refs", "missing"}
check_plan(profile, codes)        -> {"accepted", "rejected", "total_credits", "flags"}
```

- `check_plan()["accepted"]` is a list of course codes. `["rejected"]` is a list of `{"code", "reasons", "rule_refs", "missing"}`; `missing` holds unmet prerequisite codes.
- Extras that do not change the agreed shapes: `check_course()["flags"]`, `check_plan()["credit_limits"]`, `eligible_courses()`, `format_plan_result()`.
- Courses in a plan are processed in the order given: an earlier course wins over a later one that clashes with it or exceeds the credit maximum.

**RAG pipeline and chatbot (Basmala Moataz)**

```python
retrieve(query, k=5)                    -> [{"text", "source", "score", "kind", "ref"}]
recommend(profile, requested=None)      -> {"courses", "excluded", "total_credits", "sources", ...}
chat(message, history, profile)         -> reply        # profile=None starts a new-student interview
```

---

## Troubleshooting

| Problem | What to do |
|---|---|
| `ModuleNotFoundError` for a package | Run `pip install -r requirements.txt` in the same environment as the notebook kernel. |
| The advisor message looks like a template | No API key was found. Set one as shown in [LLM configuration](#llm-configuration), before section 6 of the notebook. |
| LLM error, model retired or rate limited | Backup models are tried automatically. You can also set `LLM_MODEL`, or `LLM_PROVIDER=offline` to run without an LLM. |
| The embedding model cannot be downloaded | The system falls back to TF-IDF automatically. To choose it yourself, set `ADVISOR_EMBEDDER=tfidf`. |
| Search results seem stale after editing the knowledge base | The index is rebuilt automatically when a knowledge-base file changes. To force a rebuild, delete the `index/` folder. |
| `KBValidationError` when loading the catalog | The message lists every problem found (duplicate code, missing prerequisite, bad time format, and so on). Fix the files in `knowledge_base/`. |
| `import profile` gives the wrong module in your own script | `src/profile.py` has the same name as Python's rarely used standard-library `profile` module. Put `src` first on the path, as in the examples above. The notebook and `chatbot.py` load it by file path. |
| The chat page does not open | Open http://127.0.0.1:7860 yourself, or run `python chatbot.py` from the `src` folder. |

---

## Limitations

The full discussion is in the report (section 5). In short:

- All data is fictional and small, so retrieval is easy; the system has not been tested on long official handbooks.
- Prerequisites are a plain AND list. There are no alternatives, co-requisites, minimum grades or standing requirements.
- Courses have one fixed weekly timetable; there are no sections, rooms, capacity or exam clashes.
- Auto mode builds a plan greedily by relevance and does not search all combinations.
- The LLM message can still contain wording mistakes; the structured sections (reasons, eligibility, sources) are produced by code and can be trusted.
- Chat sessions are kept in memory only, and the project has no authentication or protection of student records.

---

## Team

| Member | Role |
|---|---|
| **Mesk Khaled** | Knowledge base, student profiles, rule checking and rule tests |
| **Basmala Moataz** | RAG pipeline, recommendation output, chatbot and RAG tests |#   R A G - B a s e d - S e m e s t e r - C o u r s e - A d v i s o r  
 