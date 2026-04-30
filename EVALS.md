# Moms Verdict Evaluation Results

This document contains the evaluation strategy and the latest results of the evaluation runner for the Moms Verdict pipeline.

Because LLMs are non-deterministic, we rely heavily on mechanical evaluations to ensure the pipeline's deterministic safeguards are working correctly before deploying to production.

## Methodology

We evaluate the system using synthetic test datasets designed to trigger different edge cases. We define 12 specific assertions across these datasets and run `eval_runner.py` to validate them automatically.

## Test Cases

| ID | Name | Description |
|:---|:---|:---|
| TC01 | Happy path — large English dataset | Large genuine EN+AR dataset. Should produce high confidence positive verdict. |
| TC02 | Mixed sentiment — stroller | Mixed dataset. Should surface both positives and weight/bulk complaints. |
| TC03 | Fake review detection | Suspiciously similar reviews. Fake detector must flag this dataset. |
| TC04 | Insufficient reviews — refusal | Only 3 reviews. Pipeline must refuse and populate refusal_reason. |
| TC05 | Hallucination check | Reviews never explicitly mention BPA-free. Verdict must not hallucinate this claim. |
| TC06 | Empty reviews list | Empty input. Pipeline must return INSUFFICIENT, not crash. |
| TC07 | Borderline volume | 6 reviews — just above minimum threshold. Should produce LOW confidence verdict. |
| TC08 | Rating-text mismatch detection | All 5-star ratings but text is negative. Preprocessor should catch rating-text mismatches. |
| TC09 | Native Arabic reviews only | All Arabic reviews. Arabic verdict must be generated natively and contain Arabic text. |
| TC10 | Contradictory reviews | Reviews directly contradict on nipple flow. Should appear in both pros and cons. |
| TC11 | Very short reviews edge case | All reviews are very short (under 5 words). Pipeline should still work without crashing. |
| TC12 | High volume — theme richness | Large dataset should produce 3+ themes identified covering different topics. |

---

## Evaluation Results Run

```text
=================================================================
MOMS VERDICT — EVAL RUNNER (12 Test Cases)
=================================================================
Loaded 12 test cases.


─────────────────────────────────────────────────────────────────
[TC01] Happy path — large English dataset
Description: Large genuine EN+AR dataset. Should produce high confidence positive verdict.
Product: Philips Avent Natural Baby Bottle 260ml (51 reviews)
╭────────────────────────────────── Starting ──────────────────────────────────╮
│ Moms Verdict Pipeline                                                        │
│ Product: Philips Avent Natural Baby Bottle 260ml                             │
│ Input reviews: 51                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯

Stage 1/6 — Preprocessing...
  [Preprocessor] 51 in → 51 clean, 0 dropped, 3 rating-text mismatches
  ✓ 51 clean reviews (0 dropped, 3 rating-text mismatches)

Stage 2/6 — Fake review detection...
  [FakeDetector] Embedding 51 reviews...
  [FakeDetector] Average pairwise similarity: 0.3628
  ✓ Reviews appear genuine (similarity score: 0.363)

Stage 3/6 — Theme clustering...
  [Clusterer] Reusing 51 pre-computed embeddings.
  [Clusterer] Using 4 clusters for 51 reviews.
  [Clusterer] Final clusters:
    [1] 'Nipple flow is too fast for my newborn' — 23 reviews
    [0] 'Great bottle overall' — 11 reviews
    [2] 'سهلة التنظيف وتدخل في غسالة الأطباق بدون مشاكل' — 11 reviews
    [3] 'Absolutely worth the price' — 6 reviews
  ✓ 4 themes identified:
    • 'Nipple flow is too fast for my newborn' (23 reviews)
    • 'Great bottle overall' (11 reviews)
    • 'سهلة التنظيف وتدخل في غسالة الأطباق بدون مشاكل' (11 reviews)
    • 'Absolutely worth the price' (6 reviews)

Stage 4/6 — Extracting structured data from 4 clusters...
  [Extractor] Processing cluster 1/4: 'Nipple flow is too fast for my newborn' (23 reviews)
  [Extractor] Processing cluster 2/4: 'Great bottle overall' (11 reviews)
  [Extractor] Error attempt 1: Invalid control character at: line 1 column 183 (char 182)
  [Extractor] Waiting 15s before retry...
  [Extractor] Processing cluster 3/4: 'سهلة التنظيف وتدخل في غسالة الأطباق بدون مشاكل' (11 reviews)
  [Extractor] Processing cluster 4/4: 'Absolutely worth the price' (6 reviews)
  [Extractor] Completed. 4/4 clusters extracted.
  ✓ Extracted from 4/4 clusters

Stage 5/6 — Validating and assembling...
  [Validator] Assembling 4 cluster extractions...
  [Validator] Mismatch penalty applied: -0.012 → 0.888
  [Validator] Confidence: 0.888 (ConfidenceLevel.HIGH)
  [Validator] Final: 5 pros, 2 cons
  [Validator] Overall sentiment: OverallSentiment.MIXED
  ✓ 5 pros, 2 cons, sentiment: OverallSentiment.MIXED, confidence: 0.888 
(ConfidenceLevel.HIGH)

Stage 6/6 — Generating Arabic verdict...
  [ArabicGenerator] Generating Arabic verdict (attempt 1/3)...
  [ArabicGenerator] Done (248 Arabic chars)
  ✓ Arabic verdict generated (299 chars)

Assembling — Building final verdict...
╭──────────────────────────────────── Done ────────────────────────────────────╮
│ ✓ Verdict complete                                                           │
│ Confidence: 0.888 (ConfidenceLevel.HIGH)                                     │
│ Pros: 5 | Cons: 2 | Themes: 4                                                │
│ Fake flag: NO                                                                │
╰──────────────────────────────────────────────────────────────────────────────╯

Results:
  ✅ schema_valid: PASS (2/2)
  ✅ confidence_correct: PASS (2/2) — got high
  ✅ fake_flag: PASS (1/1) — flagged=False
  ✅ refusal_correct: PASS (1/1)
  ✅ no_hallucination: PASS (1/1) — no forbidden terms found
  ✅ arabic_valid: PASS (1/1) — 248 Arabic chars

Score: 8/8

─────────────────────────────────────────────────────────────────
[TC02] Mixed sentiment — stroller
Description: Mixed dataset. Should surface both positives and weight/bulk complaints.
Product: Graco Modes Travel System Stroller (30 reviews)
╭────────────────────────────────── Starting ──────────────────────────────────╮
│ Moms Verdict Pipeline                                                        │
│ Product: Graco Modes Travel System Stroller                                  │
│ Input reviews: 30                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯

Stage 1/6 — Preprocessing...
  [Preprocessor] 30 in → 30 clean, 0 dropped, 0 rating-text mismatches
  ✓ 30 clean reviews (0 dropped, 0 rating-text mismatches)

Stage 2/6 — Fake review detection...
  [FakeDetector] Embedding 30 reviews...
  [FakeDetector] Average pairwise similarity: 0.2773
  ✓ Reviews appear genuine (similarity score: 0.277)

Stage 3/6 — Theme clustering...
  [Clusterer] Reusing 30 pre-computed embeddings.
  [Clusterer] Using 3 clusters for 30 reviews.
  [Clusterer] Final clusters:
    [2] 'It's heavy and a bit bulky but honestly the features make up...' — 12 reviews
    [0] 'توافق مع كرسي السيارة ممتاز' — 11 reviews
    [1] 'Wheels started wobbling after 6 months of use' — 7 reviews
  ✓ 3 themes identified:
    • 'It's heavy and a bit bulky but honestly the features make up...' (12 
reviews)
    • 'توافق مع كرسي السيارة ممتاز' (11 reviews)
    • 'Wheels started wobbling after 6 months of use' (7 reviews)

Stage 4/6 — Extracting structured data from 3 clusters...
  [Extractor] Processing cluster 1/3: 'It's heavy and a bit bulky but honestly the features make up...' (12 reviews)
  [Extractor] Error attempt 1: Expecting ':' delimiter: line 1 column 11 (char 10)
  [Extractor] Waiting 15s before retry...
  [Extractor] Processing cluster 2/3: 'توافق مع كرسي السيارة ممتاز' (11 reviews)
  [Extractor] Processing cluster 3/3: 'Wheels started wobbling after 6 months of use' (7 reviews)
  [Extractor] Completed. 3/3 clusters extracted.
  ✓ Extracted from 3/3 clusters

Stage 5/6 — Validating and assembling...
  [Validator] Assembling 3 cluster extractions...
  [Validator] Confidence: 0.72 (ConfidenceLevel.MEDIUM)
  [Validator] Final: 4 pros, 4 cons
  [Validator] Overall sentiment: OverallSentiment.MIXED
  ✓ 4 pros, 4 cons, sentiment: OverallSentiment.MIXED, confidence: 0.72 
(ConfidenceLevel.MEDIUM)

Stage 6/6 — Generating Arabic verdict...
  [ArabicGenerator] Generating Arabic verdict (attempt 1/3)...
  [ArabicGenerator] Done (218 Arabic chars)
  ✓ Arabic verdict generated (261 chars)

Assembling — Building final verdict...
╭──────────────────────────────────── Done ────────────────────────────────────╮
│ ✓ Verdict complete                                                           │
│ Confidence: 0.72 (ConfidenceLevel.MEDIUM)                                    │
│ Pros: 4 | Cons: 4 | Themes: 3                                                │
│ Fake flag: NO                                                                │
╰──────────────────────────────────────────────────────────────────────────────╯

Results:
  ✅ schema_valid: PASS (2/2)
  ⚠️ confidence_correct: SKIP (not specified)
  ✅ fake_flag: PASS (1/1) — flagged=False
  ✅ refusal_correct: PASS (1/1)
  ⚠️ no_hallucination: SKIP (no forbidden terms)
  ✅ arabic_valid: PASS (1/1) — 218 Arabic chars

Score: 7/8

─────────────────────────────────────────────────────────────────
[TC03] Fake review detection
Description: Suspiciously similar reviews. Fake detector must flag this dataset.
Product: BabyBliss Premium Baby Bottle 240ml (30 reviews)
╭────────────────────────────────── Starting ──────────────────────────────────╮
│ Moms Verdict Pipeline                                                        │
│ Product: BabyBliss Premium Baby Bottle 240ml                                 │
│ Input reviews: 30                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯

Stage 1/6 — Preprocessing...
  [Preprocessor] 30 in → 30 clean, 0 dropped, 0 rating-text mismatches
  ✓ 30 clean reviews (0 dropped, 0 rating-text mismatches)

Stage 2/6 — Fake review detection...
  [FakeDetector] Embedding 30 reviews...
  [FakeDetector] Average pairwise similarity: 0.9298
  ⚠ Fake reviews detected — similarity score: 0.930. Confidence will be reduced.

Stage 3/6 — Theme clustering...
  [Clusterer] Reusing 30 pre-computed embeddings.
  [Clusterer] Using 3 clusters for 30 reviews.
  [Clusterer] Merging 1 tiny cluster(s) into nearest large cluster.
  [Clusterer] Merged cluster 1 (1 reviews) → cluster 0 'Best quality, amazing bottle'
  [Clusterer] Final clusters:
    [0] 'Best quality, amazing bottle' — 17 reviews
    [2] 'My baby loves this amazing bottle' — 13 reviews
  ✓ 2 themes identified:
    • 'Best quality, amazing bottle' (17 reviews)
    • 'My baby loves this amazing bottle' (13 reviews)

Stage 4/6 — Extracting structured data from 2 clusters...
  [Extractor] Processing cluster 1/2: 'Best quality, amazing bottle' (17 reviews)
  [Extractor] Error attempt 1: 'NoneType' object is not subscriptable
  [Extractor] Waiting 15s before retry...
  [Extractor] Processing cluster 2/2: 'My baby loves this amazing bottle' (13 reviews)
  [Extractor] Completed. 2/2 clusters extracted.
  ✓ Extracted from 2/2 clusters

Stage 5/6 — Validating and assembling...
  [Validator] Assembling 2 cluster extractions...
  [Validator] Fake flag penalty applied: -0.3 → 0.420
  [Validator] Confidence: 0.42 (ConfidenceLevel.MEDIUM)
  [Validator] Final: 5 pros, 0 cons
  [Validator] Overall sentiment: OverallSentiment.POSITIVE
  ✓ 5 pros, 0 cons, sentiment: OverallSentiment.POSITIVE, confidence: 0.42 
(ConfidenceLevel.MEDIUM)

Stage 6/6 — Generating Arabic verdict...
  [ArabicGenerator] Generating Arabic verdict (attempt 1/3)...
  [ArabicGenerator] Done (181 Arabic chars)
  ✓ Arabic verdict generated (223 chars)

Assembling — Building final verdict...
╭──────────────────────────────────── Done ────────────────────────────────────╮
│ ✓ Verdict complete                                                           │
│ Confidence: 0.42 (ConfidenceLevel.MEDIUM)                                    │
│ Pros: 5 | Cons: 0 | Themes: 2                                                │
│ Fake flag: YES                                                               │
╰──────────────────────────────────────────────────────────────────────────────╯

Results:
  ✅ schema_valid: PASS (2/2)
  ⚠️ confidence_correct: SKIP (not specified)
  ✅ fake_flag: PASS (1/1) — flagged=True
  ✅ refusal_correct: PASS (1/1)
  ⚠️ no_hallucination: SKIP (no forbidden terms)
  ✅ arabic_valid: PASS (1/1) — 181 Arabic chars

Score: 7/8

─────────────────────────────────────────────────────────────────
[TC04] Insufficient reviews — refusal
Description: Only 3 reviews. Pipeline must refuse and populate refusal_reason.
Product: NovaBaby Silicone Bottle 150ml (3 reviews)
╭────────────────────────────────── Starting ──────────────────────────────────╮
│ Moms Verdict Pipeline                                                        │
│ Product: NovaBaby Silicone Bottle 150ml                                      │
│ Input reviews: 3                                                             │
╰──────────────────────────────────────────────────────────────────────────────╯

Stage 1/6 — Preprocessing...
  [Preprocessor] 3 in → 3 clean, 0 dropped, 0 rating-text mismatches
  ✓ 3 clean reviews (0 dropped, 0 rating-text mismatches)

⚠ INSUFFICIENT DATA — only 3 reviews after preprocessing. Minimum required: 5.

Results:
  ✅ schema_valid: PASS (2/2)
  ✅ confidence_correct: PASS (2/2) — got insufficient
  ✅ fake_flag: PASS (1/1) — flagged=False
  ✅ refusal_correct: PASS (1/1)
  ⚠️ no_hallucination: SKIP (no forbidden terms)
  ⚠️ arabic_valid: SKIP (refusal case)

Score: 8/8

─────────────────────────────────────────────────────────────────
[TC05] Hallucination check — BPA-free not mentioned
Description: Reviews never explicitly mention BPA-free. Verdict must not hallucinate this claim.
Product: Philips Avent Natural Baby Bottle 260ml (51 reviews)
╭────────────────────────────────── Starting ──────────────────────────────────╮
│ Moms Verdict Pipeline                                                        │
│ Product: Philips Avent Natural Baby Bottle 260ml                             │
│ Input reviews: 51                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯

Stage 1/6 — Preprocessing...
  [Preprocessor] 51 in → 51 clean, 0 dropped, 3 rating-text mismatches
  ✓ 51 clean reviews (0 dropped, 3 rating-text mismatches)

Stage 2/6 — Fake review detection...
  [FakeDetector] Embedding 51 reviews...
  [FakeDetector] Average pairwise similarity: 0.3628
  ✓ Reviews appear genuine (similarity score: 0.363)

Stage 3/6 — Theme clustering...
  [Clusterer] Reusing 51 pre-computed embeddings.
  [Clusterer] Using 4 clusters for 51 reviews.
  [Clusterer] Final clusters:
    [1] 'Nipple flow is too fast for my newborn' — 23 reviews
    [0] 'Great bottle overall' — 11 reviews
    [2] 'سهلة التنظيف وتدخل في غسالة الأطباق بدون مشاكل' — 11 reviews
    [3] 'Absolutely worth the price' — 6 reviews
  ✓ 4 themes identified:
    • 'Nipple flow is too fast for my newborn' (23 reviews)
    • 'Great bottle overall' (11 reviews)
    • 'سهلة التنظيف وتدخل في غسالة الأطباق بدون مشاكل' (11 reviews)
    • 'Absolutely worth the price' (6 reviews)

Stage 4/6 — Extracting structured data from 4 clusters...
  [Extractor] Processing cluster 1/4: 'Nipple flow is too fast for my newborn' (23 reviews)
  [Extractor] Error attempt 1: 'NoneType' object has no attribute 'strip'
  [Extractor] Waiting 15s before retry...
  [Extractor] Processing cluster 2/4: 'Great bottle overall' (11 reviews)
  [Extractor] Processing cluster 3/4: 'سهلة التنظيف وتدخل في غسالة الأطباق بدون مشاكل' (11 reviews)
  [Extractor] Error attempt 1: 'NoneType' object has no attribute 'strip'
  [Extractor] Waiting 15s before retry...
  [Extractor] Processing cluster 4/4: 'Absolutely worth the price' (6 reviews)
  [Extractor] Completed. 4/4 clusters extracted.
  ✓ Extracted from 4/4 clusters

Stage 5/6 — Validating and assembling...
  [Validator] Assembling 4 cluster extractions...
  [Validator] Mismatch penalty applied: -0.012 → 0.888
  [Validator] Confidence: 0.888 (ConfidenceLevel.HIGH)
  [Validator] Final: 5 pros, 5 cons
  [Validator] Overall sentiment: OverallSentiment.POSITIVE
  ✓ 5 pros, 5 cons, sentiment: OverallSentiment.POSITIVE, confidence: 0.888 
(ConfidenceLevel.HIGH)

Stage 6/6 — Generating Arabic verdict...
  [ArabicGenerator] Generating Arabic verdict (attempt 1/3)...
  [ArabicGenerator] Done (203 Arabic chars)
  ✓ Arabic verdict generated (259 chars)

Assembling — Building final verdict...
╭──────────────────────────────────── Done ────────────────────────────────────╮
│ ✓ Verdict complete                                                           │
│ Confidence: 0.888 (ConfidenceLevel.HIGH)                                     │
│ Pros: 5 | Cons: 5 | Themes: 4                                                │
│ Fake flag: NO                                                                │
╰──────────────────────────────────────────────────────────────────────────────╯

Results:
  ✅ schema_valid: PASS (2/2)
  ✅ confidence_correct: PASS (2/2) — got high
  ✅ fake_flag: PASS (1/1) — flagged=False
  ✅ refusal_correct: PASS (1/1)
  ✅ no_hallucination: PASS (1/1) — no forbidden terms found
  ✅ arabic_valid: PASS (1/1) — 203 Arabic chars

Score: 8/8

─────────────────────────────────────────────────────────────────
[TC06] Empty reviews list — explicit error
Description: Empty input. Pipeline must return INSUFFICIENT, not crash.
Product: Test Product Empty (0 reviews)
╭────────────────────────────────── Starting ──────────────────────────────────╮
│ Moms Verdict Pipeline                                                        │
│ Product: Test Product Empty                                                  │
│ Input reviews: 0                                                             │
╰──────────────────────────────────────────────────────────────────────────────╯

Stage 1/6 — Preprocessing...
  [Preprocessor] 0 in → 0 clean, 0 dropped, 0 rating-text mismatches
  ✓ 0 clean reviews (0 dropped, 0 rating-text mismatches)

⚠ INSUFFICIENT DATA — only 0 reviews after preprocessing. Minimum required: 5.

Results:
  ✅ schema_valid: PASS (2/2)
  ✅ confidence_correct: PASS (2/2) — got insufficient
  ✅ fake_flag: PASS (1/1) — flagged=False
  ✅ refusal_correct: PASS (1/1)
  ⚠️ no_hallucination: SKIP (no forbidden terms)
  ⚠️ arabic_valid: SKIP (refusal case)

Score: 8/8

─────────────────────────────────────────────────────────────────
[TC07] Borderline volume — 6 reviews
Description: 6 reviews — just above minimum threshold. Should produce LOW confidence verdict.
Product: BorderlineTest Bottle 200ml (6 reviews)
╭────────────────────────────────── Starting ──────────────────────────────────╮
│ Moms Verdict Pipeline                                                        │
│ Product: BorderlineTest Bottle 200ml                                         │
│ Input reviews: 6                                                             │
╰──────────────────────────────────────────────────────────────────────────────╯

Stage 1/6 — Preprocessing...
  [Preprocessor] 6 in → 6 clean, 0 dropped, 1 rating-text mismatches
  ✓ 6 clean reviews (0 dropped, 1 rating-text mismatches)

Stage 2/6 — Fake review detection...
  [FakeDetector] Embedding 6 reviews...
  [FakeDetector] Average pairwise similarity: 0.3404
  ✓ Reviews appear genuine (similarity score: 0.340)

Stage 3/6 — Theme clustering...
  [Clusterer] Reusing 6 pre-computed embeddings.
  [Clusterer] Using 2 clusters for 6 reviews.
  [Clusterer] Merging 1 tiny cluster(s) into nearest large cluster.
  [Clusterer] Merged cluster 1 (1 reviews) → cluster 0 'Good bottle, my baby likes it a lot and no leaking issues.'
  [Clusterer] Final clusters:
    [0] 'Good bottle, my baby likes it a lot and no leaking issues.' — 6 reviews
  ✓ 1 themes identified:
    • 'Good bottle, my baby likes it a lot and no leaking issues.' (6 reviews)

Stage 4/6 — Extracting structured data from 1 clusters...
  [Extractor] Processing cluster 1/1: 'Good bottle, my baby likes it a lot and no leaking issues.' (6 reviews)
  [Extractor] Completed. 1/1 clusters extracted.
  ✓ Extracted from 1/1 clusters

Stage 5/6 — Validating and assembling...
  [Validator] Assembling 1 cluster extractions...
  [Validator] Mismatch penalty applied: -0.033 → 0.467
  [Validator] Confidence: 0.467 (ConfidenceLevel.MEDIUM)
  [Validator] Final: 0 pros, 0 cons
  [Validator] Overall sentiment: OverallSentiment.POSITIVE
  ✓ 0 pros, 0 cons, sentiment: OverallSentiment.POSITIVE, confidence: 0.467 
(ConfidenceLevel.MEDIUM)

Stage 6/6 — Generating Arabic verdict...
  [ArabicGenerator] Generating Arabic verdict (attempt 1/3)...
  [ArabicGenerator] Done (143 Arabic chars)
  ✓ Arabic verdict generated (174 chars)

Assembling — Building final verdict...
╭──────────────────────────────────── Done ────────────────────────────────────╮
│ ✓ Verdict complete                                                           │
│ Confidence: 0.467 (ConfidenceLevel.MEDIUM)                                   │
│ Pros: 0 | Cons: 0 | Themes: 1                                                │
│ Fake flag: NO                                                                │
╰──────────────────────────────────────────────────────────────────────────────╯

Results:
  ✅ schema_valid: PASS (2/2)
  ❌ confidence_correct: FAIL (0/2) — expected low, got medium
  ✅ fake_flag: PASS (1/1) — flagged=False
  ✅ refusal_correct: PASS (1/1)
  ⚠️ no_hallucination: SKIP (no forbidden terms)
  ✅ arabic_valid: PASS (1/1) — 143 Arabic chars

Score: 6/8

─────────────────────────────────────────────────────────────────
[TC08] Rating-text mismatch detection
Description: All 5-star ratings but text is negative. Preprocessor should catch rating-text mismatches.
Product: MismatchTest Baby Bottle (7 reviews)
╭────────────────────────────────── Starting ──────────────────────────────────╮
│ Moms Verdict Pipeline                                                        │
│ Product: MismatchTest Baby Bottle                                            │
│ Input reviews: 7                                                             │
╰──────────────────────────────────────────────────────────────────────────────╯

Stage 1/6 — Preprocessing...
  [Preprocessor] 7 in → 7 clean, 0 dropped, 7 rating-text mismatches
  ✓ 7 clean reviews (0 dropped, 7 rating-text mismatches)

Stage 2/6 — Fake review detection...
  [FakeDetector] Embedding 7 reviews...
  [FakeDetector] Average pairwise similarity: 0.4319
  ✓ Reviews appear genuine (similarity score: 0.432)

Stage 3/6 — Theme clustering...
  [Clusterer] Reusing 7 pre-computed embeddings.
  [Clusterer] Using 2 clusters for 7 reviews.
  [Clusterer] Final clusters:
    [1] 'Terrible product, broke after one week, complete waste of mo...' — 4 reviews
    [0] 'Disappointed with this product, it leaked and caused my baby...' — 3 reviews
  ✓ 2 themes identified:
    • 'Terrible product, broke after one week, complete waste of mo...' (4 
reviews)
    • 'Disappointed with this product, it leaked and caused my baby...' (3 
reviews)

Stage 4/6 — Extracting structured data from 2 clusters...
  [Extractor] Processing cluster 1/2: 'Terrible product, broke after one week, complete waste of mo...' (4 reviews)
  [Extractor] Processing cluster 2/2: 'Disappointed with this product, it leaked and caused my baby...' (3 reviews)
  [Extractor] Completed. 2/2 clusters extracted.
  ✓ Extracted from 2/2 clusters

Stage 5/6 — Validating and assembling...
  [Validator] Assembling 2 cluster extractions...
  [Validator] Mismatch penalty applied: -0.200 → 0.300
  [Validator] Confidence: 0.3 (ConfidenceLevel.LOW)
  [Validator] Final: 0 pros, 2 cons
  [Validator] Overall sentiment: OverallSentiment.NEGATIVE
  ✓ 0 pros, 2 cons, sentiment: OverallSentiment.NEGATIVE, confidence: 0.3 
(ConfidenceLevel.LOW)

Stage 6/6 — Generating Arabic verdict...
  [ArabicGenerator] Generating Arabic verdict (attempt 1/3)...
  [ArabicGenerator] Done (208 Arabic chars)
  ✓ Arabic verdict generated (281 chars)

Assembling — Building final verdict...
╭──────────────────────────────────── Done ────────────────────────────────────╮
│ ✓ Verdict complete                                                           │
│ Confidence: 0.3 (ConfidenceLevel.LOW)                                        │
│ Pros: 0 | Cons: 2 | Themes: 2                                                │
│ Fake flag: NO                                                                │
╰──────────────────────────────────────────────────────────────────────────────╯

Results:
  ✅ schema_valid: PASS (2/2)
  ⚠️ confidence_correct: SKIP (not specified)
  ✅ fake_flag: PASS (1/1) — flagged=False
  ✅ refusal_correct: PASS (1/1)
  ⚠️ no_hallucination: SKIP (no forbidden terms)
  ✅ arabic_valid: PASS (1/1) — 208 Arabic chars

Score: 7/8

─────────────────────────────────────────────────────────────────
[TC09] Native Arabic reviews only
Description: All Arabic reviews. Arabic verdict must be generated natively and contain Arabic text.
Product: Philips Avent زجاجة طبيعية (8 reviews)
╭────────────────────────────────── Starting ──────────────────────────────────╮
│ Moms Verdict Pipeline                                                        │
│ Product: Philips Avent زجاجة طبيعية                                          │
│ Input reviews: 8                                                             │
╰──────────────────────────────────────────────────────────────────────────────╯

Stage 1/6 — Preprocessing...
  [Preprocessor] 8 in → 8 clean, 0 dropped, 0 rating-text mismatches
  ✓ 8 clean reviews (0 dropped, 0 rating-text mismatches)

Stage 2/6 — Fake review detection...
  [FakeDetector] Embedding 8 reviews...
  [FakeDetector] Average pairwise similarity: 0.3627
  ✓ Reviews appear genuine (similarity score: 0.363)

Stage 3/6 — Theme clustering...
  [Clusterer] Reusing 8 pre-computed embeddings.
  [Clusterer] Using 2 clusters for 8 reviews.
  [Clusterer] Merging 1 tiny cluster(s) into nearest large cluster.
  [Clusterer] Merged cluster 1 (2 reviews) → cluster 0 'أفضل رضّاعة جربتها'
  [Clusterer] Final clusters:
    [0] 'أفضل رضّاعة جربتها' — 8 reviews
  ✓ 1 themes identified:
    • 'أفضل رضّاعة جربتها' (8 reviews)

Stage 4/6 — Extracting structured data from 1 clusters...
  [Extractor] Processing cluster 1/1: 'أفضل رضّاعة جربتها' (8 reviews)
  [Extractor] Completed. 1/1 clusters extracted.
  ✓ Extracted from 1/1 clusters

Stage 5/6 — Validating and assembling...
  [Validator] Assembling 1 cluster extractions...
  [Validator] Confidence: 0.5 (ConfidenceLevel.MEDIUM)
  [Validator] Final: 3 pros, 1 cons
  [Validator] Overall sentiment: OverallSentiment.POSITIVE
  ✓ 3 pros, 1 cons, sentiment: OverallSentiment.POSITIVE, confidence: 0.5 
(ConfidenceLevel.MEDIUM)

Stage 6/6 — Generating Arabic verdict...
  [ArabicGenerator] Generating Arabic verdict (attempt 1/3)...
  [ArabicGenerator] Done (193 Arabic chars)
  ✓ Arabic verdict generated (236 chars)

Assembling — Building final verdict...
╭──────────────────────────────────── Done ────────────────────────────────────╮
│ ✓ Verdict complete                                                           │
│ Confidence: 0.5 (ConfidenceLevel.MEDIUM)                                     │
│ Pros: 3 | Cons: 1 | Themes: 1                                                │
│ Fake flag: NO                                                                │
╰──────────────────────────────────────────────────────────────────────────────╯

Results:
  ✅ schema_valid: PASS (2/2)
  ❌ confidence_correct: FAIL (0/2) — expected low, got medium
  ✅ fake_flag: PASS (1/1) — flagged=False
  ✅ refusal_correct: PASS (1/1)
  ⚠️ no_hallucination: SKIP (no forbidden terms)
  ✅ arabic_valid: PASS (1/1) — 193 Arabic chars

Score: 6/8

─────────────────────────────────────────────────────────────────
[TC10] Contradictory reviews on same topic
Description: Reviews directly contradict on nipple flow. Should appear in both pros and cons.
Product: ContradictTest Bottle 250ml (8 reviews)
╭────────────────────────────────── Starting ──────────────────────────────────╮
│ Moms Verdict Pipeline                                                        │
│ Product: ContradictTest Bottle 250ml                                         │
│ Input reviews: 8                                                             │
╰──────────────────────────────────────────────────────────────────────────────╯

Stage 1/6 — Preprocessing...
  [Preprocessor] 8 in → 8 clean, 0 dropped, 0 rating-text mismatches
  ✓ 8 clean reviews (0 dropped, 0 rating-text mismatches)

Stage 2/6 — Fake review detection...
  [FakeDetector] Embedding 8 reviews...
  [FakeDetector] Average pairwise similarity: 0.3587
  ✓ Reviews appear genuine (similarity score: 0.359)

Stage 3/6 — Theme clustering...
  [Clusterer] Reusing 8 pre-computed embeddings.
  [Clusterer] Using 2 clusters for 8 reviews.
  [Clusterer] Final clusters:
    [0] 'The nipple flow is perfect, not too fast not too slow, ideal...' — 4 reviews
    [1] 'Flow is far too fast, milk pours out and baby cannot keep up...' — 4 reviews
  ✓ 2 themes identified:
    • 'The nipple flow is perfect, not too fast not too slow, ideal...' (4 
reviews)
    • 'Flow is far too fast, milk pours out and baby cannot keep up...' (4 
reviews)

Stage 4/6 — Extracting structured data from 2 clusters...
  [Extractor] Processing cluster 1/2: 'The nipple flow is perfect, not too fast not too slow, ideal...' (4 reviews)
  [Extractor] Processing cluster 2/2: 'Flow is far too fast, milk pours out and baby cannot keep up...' (4 reviews)
  [Extractor] Error attempt 1: 'NoneType' object has no attribute 'strip'
  [Extractor] Waiting 15s before retry...
  [Extractor] Completed. 2/2 clusters extracted.
  ✓ Extracted from 2/2 clusters

Stage 5/6 — Validating and assembling...
  [Validator] Assembling 2 cluster extractions...
  [Validator] Confidence: 0.5 (ConfidenceLevel.MEDIUM)
  [Validator] Final: 1 pros, 1 cons
  [Validator] Overall sentiment: OverallSentiment.MIXED
  ✓ 1 pros, 1 cons, sentiment: OverallSentiment.MIXED, confidence: 0.5 
(ConfidenceLevel.MEDIUM)

Stage 6/6 — Generating Arabic verdict...
  [ArabicGenerator] Generating Arabic verdict (attempt 1/3)...
  [ArabicGenerator] Done (176 Arabic chars)
  ✓ Arabic verdict generated (245 chars)

Assembling — Building final verdict...
╭──────────────────────────────────── Done ────────────────────────────────────╮
│ ✓ Verdict complete                                                           │
│ Confidence: 0.5 (ConfidenceLevel.MEDIUM)                                     │
│ Pros: 1 | Cons: 1 | Themes: 2                                                │
│ Fake flag: NO                                                                │
╰──────────────────────────────────────────────────────────────────────────────╯

Results:
  ✅ schema_valid: PASS (2/2)
  ⚠️ confidence_correct: SKIP (not specified)
  ✅ fake_flag: PASS (1/1) — flagged=False
  ✅ refusal_correct: PASS (1/1)
  ⚠️ no_hallucination: SKIP (no forbidden terms)
  ✅ arabic_valid: PASS (1/1) — 176 Arabic chars

Score: 7/8

─────────────────────────────────────────────────────────────────
[TC11] Very short reviews edge case
Description: All reviews are very short (under 5 words). Pipeline should still work without crashing.
Product: ShortReview Baby Bottle 180ml (7 reviews)
╭────────────────────────────────── Starting ──────────────────────────────────╮
│ Moms Verdict Pipeline                                                        │
│ Product: ShortReview Baby Bottle 180ml                                       │
│ Input reviews: 7                                                             │
╰──────────────────────────────────────────────────────────────────────────────╯

Stage 1/6 — Preprocessing...
  [Preprocessor] 7 in → 7 clean, 0 dropped, 0 rating-text mismatches
  ✓ 7 clean reviews (0 dropped, 0 rating-text mismatches)

Stage 2/6 — Fake review detection...
  [FakeDetector] Embedding 7 reviews...
  [FakeDetector] Average pairwise similarity: 0.2863
  ✓ Reviews appear genuine (similarity score: 0.286)

Stage 3/6 — Theme clustering...
  [Clusterer] Reusing 7 pre-computed embeddings.
  [Clusterer] Using 2 clusters for 7 reviews.
  [Clusterer] Merging 1 tiny cluster(s) into nearest large cluster.
  [Clusterer] Merged cluster 1 (2 reviews) → cluster 0 'Not bad.'
  [Clusterer] Final clusters:
    [0] 'Not bad.' — 7 reviews
  ✓ 1 themes identified:
    • 'Not bad.' (7 reviews)

Stage 4/6 — Extracting structured data from 1 clusters...
  [Extractor] Processing cluster 1/1: 'Not bad.' (7 reviews)
  [Extractor] Error attempt 1: 'NoneType' object has no attribute 'strip'
  [Extractor] Waiting 15s before retry...
  [Extractor] Completed. 1/1 clusters extracted.
  ✓ Extracted from 1/1 clusters

Stage 5/6 — Validating and assembling...
  [Validator] Assembling 1 cluster extractions...
  [Validator] Confidence: 0.5 (ConfidenceLevel.MEDIUM)
  [Validator] Final: 0 pros, 0 cons
  [Validator] Overall sentiment: OverallSentiment.POSITIVE
  ✓ 0 pros, 0 cons, sentiment: OverallSentiment.POSITIVE, confidence: 0.5 
(ConfidenceLevel.MEDIUM)

Stage 6/6 — Generating Arabic verdict...
  [ArabicGenerator] Generating Arabic verdict (attempt 1/3)...
  [ArabicGenerator] Done (137 Arabic chars)
  ✓ Arabic verdict generated (196 chars)

Assembling — Building final verdict...
╭──────────────────────────────────── Done ────────────────────────────────────╮
│ ✓ Verdict complete                                                           │
│ Confidence: 0.5 (ConfidenceLevel.MEDIUM)                                     │
│ Pros: 0 | Cons: 0 | Themes: 1                                                │
│ Fake flag: NO                                                                │
╰──────────────────────────────────────────────────────────────────────────────╯

Results:
  ✅ schema_valid: PASS (2/2)
  ❌ confidence_correct: FAIL (0/2) — expected low, got medium
  ✅ fake_flag: PASS (1/1) — flagged=False
  ✅ refusal_correct: PASS (1/1)
  ⚠️ no_hallucination: SKIP (no forbidden terms)
  ✅ arabic_valid: PASS (1/1) — 137 Arabic chars

Score: 6/8

─────────────────────────────────────────────────────────────────
[TC12] High volume — theme richness
Description: Large dataset should produce 3+ themes identified covering different topics.
Product: Philips Avent Natural Baby Bottle 260ml (51 reviews)
╭────────────────────────────────── Starting ──────────────────────────────────╮
│ Moms Verdict Pipeline                                                        │
│ Product: Philips Avent Natural Baby Bottle 260ml                             │
│ Input reviews: 51                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯

Stage 1/6 — Preprocessing...
  [Preprocessor] 51 in → 51 clean, 0 dropped, 3 rating-text mismatches
  ✓ 51 clean reviews (0 dropped, 3 rating-text mismatches)

Stage 2/6 — Fake review detection...
  [FakeDetector] Embedding 51 reviews...
  [FakeDetector] Average pairwise similarity: 0.3628
  ✓ Reviews appear genuine (similarity score: 0.363)

Stage 3/6 — Theme clustering...
  [Clusterer] Reusing 51 pre-computed embeddings.
  [Clusterer] Using 4 clusters for 51 reviews.
  [Clusterer] Final clusters:
    [1] 'Nipple flow is too fast for my newborn' — 23 reviews
    [0] 'Great bottle overall' — 11 reviews
    [2] 'سهلة التنظيف وتدخل في غسالة الأطباق بدون مشاكل' — 11 reviews
    [3] 'Absolutely worth the price' — 6 reviews
  ✓ 4 themes identified:
    • 'Nipple flow is too fast for my newborn' (23 reviews)
    • 'Great bottle overall' (11 reviews)
    • 'سهلة التنظيف وتدخل في غسالة الأطباق بدون مشاكل' (11 reviews)
    • 'Absolutely worth the price' (6 reviews)

Stage 4/6 — Extracting structured data from 4 clusters...
  [Extractor] Processing cluster 1/4: 'Nipple flow is too fast for my newborn' (23 reviews)
  [Extractor] Processing cluster 2/4: 'Great bottle overall' (11 reviews)
  [Extractor] Error attempt 1: Invalid \escape: line 1 column 7 (char 6)
  [Extractor] Waiting 15s before retry...
  [Extractor] Processing cluster 3/4: 'سهلة التنظيف وتدخل في غسالة الأطباق بدون مشاكل' (11 reviews)
  [Extractor] Error attempt 1: Expecting ':' delimiter: line 1 column 32 (char 31)
  [Extractor] Waiting 15s before retry...
  [Extractor] Processing cluster 4/4: 'Absolutely worth the price' (6 reviews)
  [Extractor] Completed. 4/4 clusters extracted.
  ✓ Extracted from 4/4 clusters

Stage 5/6 — Validating and assembling...
  [Validator] Assembling 4 cluster extractions...
  [Validator] Mismatch penalty applied: -0.012 → 0.888
  [Validator] Confidence: 0.888 (ConfidenceLevel.HIGH)
  [Validator] Final: 5 pros, 2 cons
  [Validator] Overall sentiment: OverallSentiment.MIXED
  ✓ 5 pros, 2 cons, sentiment: OverallSentiment.MIXED, confidence: 0.888 
(ConfidenceLevel.HIGH)

Stage 6/6 — Generating Arabic verdict...
  [ArabicGenerator] Generating Arabic verdict (attempt 1/3)...
  [ArabicGenerator] Done (273 Arabic chars)
  ✓ Arabic verdict generated (331 chars)

Assembling — Building final verdict...
╭──────────────────────────────────── Done ────────────────────────────────────╮
│ ✓ Verdict complete                                                           │
│ Confidence: 0.888 (ConfidenceLevel.HIGH)                                     │
│ Pros: 5 | Cons: 2 | Themes: 4                                                │
│ Fake flag: NO                                                                │
╰──────────────────────────────────────────────────────────────────────────────╯

Results:
  ✅ schema_valid: PASS (2/2)
  ✅ confidence_correct: PASS (2/2) — got high
  ✅ fake_flag: PASS (1/1) — flagged=False
  ✅ refusal_correct: PASS (1/1)
  ⚠️ no_hallucination: SKIP (no forbidden terms)
  ✅ arabic_valid: PASS (1/1) — 273 Arabic chars

Score: 8/8

=================================================================
SCORECARD
=================================================================
  [TC01] ████████ 8/8 — Happy path — large English dataset
  [TC02] ███████░ 7/8 — Mixed sentiment — stroller
  [TC03] ███████░ 7/8 — Fake review detection
  [TC04] ████████ 8/8 — Insufficient reviews — refusal
  [TC05] ████████ 8/8 — Hallucination check — BPA-free not mentioned
  [TC06] ████████ 8/8 — Empty reviews list — explicit error
  [TC07] ██████░░ 6/8 — Borderline volume — 6 reviews
  [TC08] ███████░ 7/8 — Rating-text mismatch detection
  [TC09] ██████░░ 6/8 — Native Arabic reviews only
  [TC10] ███████░ 7/8 — Contradictory reviews on same topic
  [TC11] ██████░░ 6/8 — Very short reviews edge case
  [TC12] ████████ 8/8 — High volume — theme richness

Overall: 86/96 (89.6%)
Rating: ✅ STRONG
=================================================================
```
