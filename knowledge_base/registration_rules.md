# Registration Rules (FICTIONAL sample policy)

> **FICTIONAL SAMPLE POLICY.** These rules were written for this assignment. They are not taken from any official university document and do not describe any real institution. Every rule has a fixed ID (R1 to R9) that the rule checker (`src/rules.py`) uses when it accepts, rejects, or flags a course.

## Terms

- **Credit hours** are the credits of a course as listed in `courses.json`.
- **Level** is the student's year of study (1 to 4). A course has a level in the catalog.
- **GPA** is the student's cumulative GPA on a 4.00 scale.
- **Regular semester** means Fall or Spring.
- **Passed course** means a course the student completed successfully before the semester starts.

### R1 - Credit-hour range

In a regular semester a student must register for at least 12 and at most 18 credit hours. A semester load below 12 credit hours is not allowed unless rule R3 applies. The limits of R2 and R3 replace the maximum of 18 for the students they describe.

### R2 - Overload for high GPA

A student with a cumulative GPA of 3.50 or higher may register for up to 21 credit hours. The minimum of 12 credit hours from R1 still applies. A GPA below 3.50 gives no overload, and the maximum stays at 18 credit hours.

### R3 - Academic probation

A student with a cumulative GPA below 2.00 is on academic probation. A student on probation must register for at least 9 and at most 12 credit hours. R3 replaces both R1 and R2 for this student.

### R4 - Level eligibility

A student may register only for courses whose level is equal to or lower than the student's own level. A student in level 3 may take level 1, 2, and 3 courses, but not level 4 courses. There are no exceptions to this rule.

### R5 - Prerequisite policy

Every prerequisite listed for a course must be a passed course before the semester starts. A prerequisite cannot be taken at the same time as the course. Chained prerequisites apply in order, so a student must pass each course in the chain before the next one.

### R6 - Offered-semester rule

A course may be registered only in a semester in which it is offered. The semesters in which a course is offered are listed in the catalog.

### R7 - No time clashes

A student may not register for two courses whose meeting times overlap on the same day. Two meetings clash if one starts before the other ends. A meeting that ends exactly when another starts is not a clash.

### R8 - No retaking passed courses

A student may not register again for a course that is already in the student's list of passed courses.

### R9 - Missing or unknown information

If a required item is missing from the student record (level, GPA, semester, or the list of passed courses), or if a course code is not in the catalog, the advisor must flag the problem. The advisor must not guess the missing value. A course whose eligibility depends on missing information is not recommended until the information is supplied.
