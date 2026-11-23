# Expected results for the six test profiles

Hand-derived from `courses.json` and `registration_rules.md` (FICTIONAL sample policy). Each profile is checked with `check_plan(profile, candidate_plan)`; the tests in `tests/test_rules.py` compare the real output with `expected_results.json`, which holds the same data in machine-readable form.

Credit limits: R1 = 12-18, R2 (GPA >= 3.50) = 12-21, R3 (GPA < 2.00, probation) = 9-12.

## S01 - Normal strong student (level 3, GPA 3.82, Fall)

- **Profile file:** `profiles/student_01.json`
- **Candidate plan (in priority order):** CS301, CS302, CS205, CS307
- **Credit limits that apply:** 12-21 (R2)
- **Expected accepted:** CS301, CS302, CS205, CS307  (total 13 credit hours)
- **Expected rejected:** none
- **Expected flags:** none
- **Why this profile:** Every prerequisite is passed, nothing clashes (adjacent slots such as CS205 and CS301 only touch), and 13 credits is inside 12-21.

## S02 - UNMET PREREQUISITE (level 3, GPA 2.9, Fall) - MATH202 never passed

- **Profile file:** `profiles/student_02.json`
- **Candidate plan (in priority order):** CS301, CS304, CS302, CS205, CS307, CS402
- **Credit limits that apply:** 12-18 (R1)
- **Expected accepted:** CS304, CS302, CS205, CS307  (total 13 credit hours)
- **Expected rejected:**
  - `CS301` - rule(s) R5; missing MATH202: needs CS201, MATH201 and MATH202; MATH202 is not passed
  - `CS402` - rule(s) R4, R5; missing CS303: level 4 course for a level 3 student, and prerequisite CS303 is not passed
- **Expected flags:** none
- **Why this profile:** CS304 (computer vision) is accepted instead of CS301 because its prerequisites CS201 and MATH201 are passed.

## S03 - Low-GPA probation student (level 3, GPA 1.8, Fall)

- **Profile file:** `profiles/student_03.json`
- **Candidate plan (in priority order):** CS301, CS302, CS205, CS307, CS304
- **Credit limits that apply:** 9-12 (R3)
- **Expected accepted:** CS301, CS302, CS205  (total 9 credit hours)
- **Expected rejected:**
  - `CS307` - rule(s) R3: 9 + 4 credits = 13, above the probation maximum of 12
  - `CS304` - rule(s) R7: same time slot as the accepted CS301 (Sun and Tue 12:30-14:00)
- **Expected flags:** 1 (R3), mentioning: desired credit load of 15
- **Why this profile:** Same history and same plan as S01, but the probation limit of 12 credits changes the outcome. Desired load 15 is flagged.

## S04 - CREDIT LIMIT EXCEEDED (level 4, GPA 3.2, Fall, wants 21 credits)

- **Profile file:** `profiles/student_04.json`
- **Candidate plan (in priority order):** CS402, CS404, CS490, CS307, CS304, HUM401, CS204
- **Credit limits that apply:** 12-18 (R1)
- **Expected accepted:** CS402, CS404, CS490, CS307, CS304, HUM401  (total 18 credit hours)
- **Expected rejected:**
  - `CS204` - rule(s) R1: 18 + 3 credits = 21, above the maximum of 18; overload needs GPA >= 3.50 (R2) and the student has 3.2
- **Expected flags:** 1 (R1), mentioning: desired credit load of 21
- **Why this profile:** The first six courses fill exactly 18 credits, so the seventh is rejected. Desired load 21 is flagged.

## S05 - TIME CLASH (level 4, GPA 3.6, Spring)

- **Profile file:** `profiles/student_05.json`
- **Candidate plan (in priority order):** CS305, CS306, CS303, CS310, CS401, CS403, HUM401
- **Credit limits that apply:** 12-21 (R2)
- **Expected accepted:** CS305, CS303, CS310, CS401, CS403, HUM401  (total 17 credit hours)
- **Expected rejected:**
  - `CS306` - rule(s) R7: partial overlap with CS305: Tue 13:30-15:00 vs Tue 12:30-14:00
- **Expected flags:** none
- **Why this profile:** Tests partial overlap (not just identical slots).

## S06 - MISSING DATA (level missing, desired_credits missing, unknown passed code CS999)

- **Profile file:** `profiles/student_06.json`
- **Candidate plan (in priority order):** CS101, CS201, CS204, XYZ123
- **Credit limits that apply:** 12-18 (R1)
- **Expected accepted:** none  (total 0 credit hours)
- **Expected rejected:**
  - `CS101` - rule(s) R8, R9: already passed (R8), and level cannot be verified (R9)
  - `CS201` - rule(s) R9: level missing, so R4 cannot be verified; prerequisite CS102 is passed
  - `CS204` - rule(s) R9: level missing, so R4 cannot be verified; prerequisite CS102 is passed
  - `XYZ123` - rule(s) R9: course code is not in the catalog
- **Expected flags:** 4 (R9, R9, R9, R1), mentioning: 'level'; 'desired_credits'; CS999; below the minimum
- **Why this profile:** Nothing is approved on a guess: with the level missing, no course can be accepted.
