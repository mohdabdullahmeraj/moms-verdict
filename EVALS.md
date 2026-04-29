# Moms Verdict Evaluation Results

This document contains the evaluation strategy and the latest results of the evaluation runner for the Moms Verdict pipeline. 

Because LLMs are non-deterministic, we rely heavily on mechanical evaluations to ensure the pipeline's deterministic safeguards are working correctly before deploying to production.

## Methodology

We evaluate the system using 4 synthetic test datasets designed to trigger different edge cases:
1. **Philips Avent (Genuine, Large, Mixed Languages)**: Tests standard behavior and Arabic generation capability.
2. **Graco Stroller (Genuine, Medium)**: Tests behavior on smaller sample sizes.
3. **BabyBliss (Spam)**: Contains identical, repeated fake reviews. Tests the mechanical `FakeDetector`.
4. **NovaBaby (Sparse)**: Contains < 5 reviews. Tests the mechanical insufficient data gate.

We define 12 assertions across these datasets and run `eval_runner.py` to validate them automatically.

## Test Cases

| ID | Target Product | Assertion | Reason |
|:---|:---|:---|:---|
| TC-01 | NovaBaby Carrier | Confidence == INSUFFICIENT | The pipeline must refuse to provide a verdict for < 5 reviews. |
| TC-02 | NovaBaby Carrier | Fake review flag == False | The fake review check should be skipped for sparse data. |
| TC-03 | NovaBaby Carrier | Verdicts == Empty | The LLM should not be called at all for sparse data. |
| TC-04 | BabyBliss Bouncer | Fake flag == True | The FakeDetector MUST catch spam using similarity embeddings. |
| TC-05 | BabyBliss Bouncer | Similarity > 0.85 | Validates the O(n^2) cosine similarity matrix works. |
| TC-06 | BabyBliss Bouncer | Confidence Score < 0.50 | Fake flags must penalize the final confidence score mathematically. |
| TC-07 | Philips Avent | Confidence Level == HIGH | A genuine product with 120 reviews should earn HIGH confidence. |
| TC-08 | Philips Avent | Sentiment == POSITIVE | Aggregated sentiment should correctly reflect the data. |
| TC-09 | Philips Avent | Valid Arabic Characters | Arabic generation must actually return Unicode Arabic, not English. |
| TC-10 | Graco Stroller | Confidence in [MEDIUM, HIGH] | Proper confidence tiering for medium datasets. |
| TC-11 | ALL | Quotes Exist | Grounding check: Every extracted pro/con MUST have a verbatim quote. |
| TC-12 | ALL | Max 5 Pros/Cons | Ensures we do not overflow the UI design limits. |

---

## Evaluation Results Run

*(Run `python evals/eval_runner.py` in Colab and paste the console output here)*
