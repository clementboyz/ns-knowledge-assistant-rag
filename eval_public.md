# Evaluation (Public Docs)

Goal: Verify retrieval quality and citation correctness on the public sample corpus (`docs_public/`).

## Test Questions
1. What are the steps to borrow equipment?
2. What are the steps to return equipment?
3. What should you do if an item is damaged?
4. Which fields must be recorded during borrowing?
5. What does the admin verify during borrowing?
6. Where are restricted/unit documents stored?
7. Where are public demo documents stored?
8. What is the purpose of citations in this assistant?
9. How does the assistant decide which text to show?
10. What is the limitation of this MVP version?

## Expected Behavior
- The top retrieved chunk should come from the most relevant doc (e.g., `docs_public/sop_sample.md` for borrow/return questions).
- Citations should include the correct source file and chunk id.
- If the answer is not in the public docs, the assistant should retrieve irrelevant/weak matches — this indicates the need for more documents or a synthesis layer.
