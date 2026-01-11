# Day 3 Evaluation (Hybrid Retrieval)

Goal: check whether hybrid retrieval helps match different phrasing styles.

PASS rule:
- Top-1 evidence is the correct doc AND contains the answer.

| # | Question | Expected doc | Result | Notes |
|---|---|---|---|---|
| 1 | What info must be recorded when borrowing? | faq_sample.md | PASS | Extracted list matches Q2 |
| 2 | What are the steps to borrow equipment? | sop_sample.md |  |  |
| 3 | What are the steps to return equipment? | sop_sample.md |  |  |
| 4 | What should you do if an item is damaged? | sop_sample.md / checklist_sample.md |  |  |
| 5 | How do we reduce disputes about condition? | faq_sample.md |  |  |
| 6 | What if the system is down? | faq_sample.md |  |  |
| 7 | Can someone else return the item? | faq_sample.md |  |  |
