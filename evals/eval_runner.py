"""
evals/eval_runner.py
Runs all 12 eval test cases and prints a scorecard.
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MomsVerdictPipeline
from src.schema import MomsVerdict, ConfidenceLevel

SAMPLE_DIR = Path("data/sample_products")
CASES_FILE = Path("evals/test_cases.json")
MAX_SCORE_PER_CASE = 8


def count_arabic(text):
    return sum(1 for c in text if '\u0600' <= c <= '\u06FF')


def run_eval(case, verdict):
    details = {}
    score = 0

    # Schema validity (2 pts)
    try:
        _ = verdict.model_dump_json()
        details["schema_valid"] = "PASS (2/2)"
        score += 2
    except Exception as e:
        details["schema_valid"] = f"FAIL (0/2) — {e}"

    # Confidence level correct (2 pts)
    expected_conf = case.get("expected_confidence_level")
    if expected_conf is None:
        details["confidence_correct"] = "SKIP (not specified)"
        score += 1
    elif verdict.confidence_level.value == expected_conf:
        details["confidence_correct"] = f"PASS (2/2) — got {verdict.confidence_level.value}"
        score += 2
    else:
        details["confidence_correct"] = (
            f"FAIL (0/2) — expected {expected_conf}, "
            f"got {verdict.confidence_level.value}"
        )

    # Fake flag correct (1 pt)
    expected_fake = case.get("expected_fake_flagged", False)
    actual_fake = verdict.fake_review_flag.flagged
    if actual_fake == expected_fake:
        details["fake_flag"] = f"PASS (1/1) — flagged={actual_fake}"
        score += 1
    else:
        details["fake_flag"] = (
            f"FAIL (0/1) — expected flagged={expected_fake}, "
            f"got flagged={actual_fake}"
        )

    # Refusal correct (1 pt)
    expected_refusal = case.get("expected_refusal", False)
    actual_refusal = verdict.confidence_level == ConfidenceLevel.INSUFFICIENT
    if actual_refusal == expected_refusal:
        details["refusal_correct"] = "PASS (1/1)"
        score += 1
    else:
        details["refusal_correct"] = (
            f"FAIL (0/1) — expected refusal={expected_refusal}, "
            f"got refusal={actual_refusal}"
        )

    # No hallucination (1 pt)
    must_not_mention = case.get("must_not_mention", [])
    if not must_not_mention:
        details["no_hallucination"] = "SKIP (no forbidden terms)"
        score += 1
    else:
        verdict_text = (
            verdict.verdict_en + " " +
            " ".join(p.point for p in verdict.pros) + " " +
            " ".join(c.point for c in verdict.cons)
        ).lower()
        violations = [t for t in must_not_mention if t.lower() in verdict_text]
        if not violations:
            details["no_hallucination"] = "PASS (1/1) — no forbidden terms found"
            score += 1
        else:
            details["no_hallucination"] = f"FAIL (0/1) — found: {violations}"

    # Arabic valid (1 pt)
    if verdict.confidence_level == ConfidenceLevel.INSUFFICIENT:
        details["arabic_valid"] = "SKIP (refusal case)"
        score += 1
    elif count_arabic(verdict.verdict_ar) >= 10:
        details["arabic_valid"] = (
            f"PASS (1/1) — {count_arabic(verdict.verdict_ar)} Arabic chars"
        )
        score += 1
    else:
        details["arabic_valid"] = (
            f"FAIL (0/1) — only {count_arabic(verdict.verdict_ar)} Arabic chars"
        )

    return score, MAX_SCORE_PER_CASE, details


def main():
    print("=" * 65)
    print("MOMS VERDICT — EVAL RUNNER (12 Test Cases)")
    print("=" * 65)

    with open(CASES_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} test cases.\n")
    pipeline = MomsVerdictPipeline()
    results = []

    for case in cases:
        print(f"\n{'─' * 65}")
        print(f"[{case['id']}] {case['name']}")
        print(f"Description: {case['description']}")

        # load reviews — from file or inline
        if case.get("file"):
            filepath = SAMPLE_DIR / case["file"]
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                product_name = data["product_name"]
                raw_reviews = data["reviews"]
            except FileNotFoundError:
                print(f"  ❌ File not found: {filepath}")
                results.append({
                    "id": case["id"], "name": case["name"],
                    "score": 0, "max": MAX_SCORE_PER_CASE,
                    "error": "File not found"
                })
                continue
        elif "inline_reviews" in case:
            product_name = case["inline_product"]
            raw_reviews = case["inline_reviews"]
        else:
            print(f"  ❌ No reviews source specified")
            continue

        print(f"Product: {product_name} ({len(raw_reviews)} reviews)")

        # run pipeline
        try:
            verdict = pipeline.run(product_name, raw_reviews)
        except Exception as e:
            print(f"  ❌ Pipeline error: {e}")
            results.append({
                "id": case["id"], "name": case["name"],
                "score": 0, "max": MAX_SCORE_PER_CASE,
                "error": str(e)
            })
            continue

        # score
        score, max_score, details = run_eval(case, verdict)

        print(f"\nResults:")
        for criterion, result in details.items():
            icon = "✅" if "PASS" in result else ("⚠️" if "SKIP" in result else "❌")
            print(f"  {icon} {criterion}: {result}")
        print(f"\nScore: {score}/{max_score}")

        results.append({
            "id": case["id"],
            "name": case["name"],
            "score": score,
            "max": max_score,
            "details": details
        })

    # scorecard
    print(f"\n{'=' * 65}")
    print("SCORECARD")
    print(f"{'=' * 65}")
    total_score = sum(r["score"] for r in results)
    total_max = sum(r.get("max", MAX_SCORE_PER_CASE) for r in results)

    for r in results:
        bar = "█" * r["score"] + "░" * (r.get("max", MAX_SCORE_PER_CASE) - r["score"])
        print(f"  [{r['id']}] {bar} {r['score']}/{r.get('max', MAX_SCORE_PER_CASE)} — {r['name']}")

    pct = round((total_score / total_max) * 100, 1) if total_max > 0 else 0
    print(f"\nOverall: {total_score}/{total_max} ({pct}%)")

    if pct >= 80:
        print("Rating: ✅ STRONG")
    elif pct >= 60:
        print("Rating: ⚠️  ACCEPTABLE")
    else:
        print("Rating: ❌ NEEDS WORK")

    print("=" * 65)

    # save
    results_path = Path("evals/results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
