"""
data/generate_reviews.py

Synthetic review generator for Moms Verdict pipeline testing.

Generates realistic customer reviews for 3 products:
1. Philips Avent Natural Bottle     — large dataset, mixed EN/AR, genuine
2. Graco Modes Travel System        — medium dataset, mostly EN, genuine
3. Generic Brand Baby Bottle        — small dataset, fake/spam reviews
   (used to test fake review detection)

Each product gets saved as a separate JSON file in data/sample_products/.

Run this once before running the pipeline or evals:
    docker compose --profile datagen run generate-data
    OR
    python data/generate_reviews.py
"""

import json
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# use flash for data generation — cheap, fast, good enough
model = genai.GenerativeModel(os.environ.get("GEMINI_FLASH_MODEL", "gemini-1.5-flash"))

OUTPUT_DIR = Path(__file__).parent / "sample_products"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, max_retries: int = 3) -> str:
    """
    Call Gemini with retry logic.
    Free tier has rate limits — we wait between calls.
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 10
                print(f"  API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def parse_json_response(raw: str) -> list:
    """
    Strip markdown fences if present and parse JSON.
    Gemini sometimes wraps JSON in ```json ... ``` even when told not to.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # remove opening fence (```json or ```)
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned.strip())


# ---------------------------------------------------------------------------
# Product 1: Philips Avent Natural Bottle
# Large dataset, mix of EN and AR, genuine reviews
# Tests: happy path, multilingual, theme clustering, contradiction handling
# ---------------------------------------------------------------------------

def generate_avent_bottle_reviews() -> dict:
    print("\n[1/3] Generating Philips Avent Natural Bottle reviews...")

    # --- English reviews ---
    print("  Generating 80 English reviews...")
    en_prompt = """
Generate exactly 80 realistic customer reviews for the "Philips Avent Natural Baby Bottle 260ml"
sold on a Middle Eastern baby e-commerce platform.

Make them realistic and varied:
- 55 positive reviews (ratings 4-5): mention ease of cleaning, anti-colic effect,
  wide neck, good for breastfed babies, dishwasher safe
- 15 mixed reviews (ratings 3): mention it's good overall but nipple flow is fast,
  or that it's pricey for what it is
- 10 negative reviews (ratings 1-2): mention nipple flow too fast for newborns,
  leaking issues, or baby rejecting the nipple

Add realistic noise:
- Some reviews are very short (1-2 sentences)
- Some are long and detailed (5-6 sentences)
- Some have typos or informal language
- 3 reviews should have a rating that contradicts the text
  (e.g., 5 stars but the text complains)

Return ONLY a JSON array, no markdown, no explanation.
Each object must have exactly these fields:
{
  "text": "review text here",
  "rating": 4.0,
  "language": "en",
  "reviewer_name": "Sarah M."
}
"""
    en_raw = call_gemini(en_prompt)
    en_reviews = parse_json_response(en_raw)
    print(f"  Got {len(en_reviews)} English reviews")
    time.sleep(5)  # respect free tier rate limit

    # --- Arabic reviews ---
    print("  Generating 40 Arabic reviews...")
    ar_prompt = """
أنشئ بالضبط 40 تقييماً واقعياً من العملاء لمنتج "زجاجة فيليبس أفنت الطبيعية 260 مل"
على منصة تجارة إلكترونية للأطفال في منطقة الشرق الأوسط.

اجعلها واقعية ومتنوعة:
- 28 تقييماً إيجابياً (تقييم 4-5): تذكر سهولة التنظيف، وتأثير مكافحة المغص،
  والعنق الواسع، ومناسبة للرضاعة الطبيعية
- 8 تقييمات متوسطة (تقييم 3): تذكر أنها جيدة بشكل عام لكن تدفق الحليب سريع
- 4 تقييمات سلبية (تقييم 1-2): تذكر أن تدفق الحليب سريع جداً للمواليد الجدد

أضف تنوعاً واقعياً:
- بعض التقييمات قصيرة جداً (جملة أو جملتان)
- بعضها مفصل (4-5 جمل)
- استخدم اللهجة الخليجية أحياناً

أعد فقط مصفوفة JSON، بدون markdown، بدون شرح.
كل كائن يجب أن يحتوي على هذه الحقول بالضبط:
{
  "text": "نص التقييم هنا",
  "rating": 4.0,
  "language": "ar",
  "reviewer_name": "أم سارة"
}
"""
    ar_raw = call_gemini(ar_prompt)
    ar_reviews = parse_json_response(ar_raw)
    print(f"  Got {len(ar_reviews)} Arabic reviews")
    time.sleep(5)

    all_reviews = en_reviews + ar_reviews

    product = {
        "product_name": "Philips Avent Natural Baby Bottle 260ml",
        "product_id": "avent-natural-260ml",
        "category": "Feeding",
        "expected_verdict": "positive",
        "expected_confidence": "high",
        "notes": "Large genuine dataset. Should produce high confidence positive verdict with mixed notes on nipple flow.",
        "reviews": all_reviews
    }

    path = OUTPUT_DIR / "avent_bottle.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(product, f, ensure_ascii=False, indent=2)

    print(f"  Saved {len(all_reviews)} reviews to {path}")
    return product


# ---------------------------------------------------------------------------
# Product 2: Graco Modes Travel System
# Medium dataset, mostly English, genuine, more mixed sentiment
# Tests: mixed sentiment, theme variety, medium confidence
# ---------------------------------------------------------------------------

def generate_graco_stroller_reviews() -> dict:
    print("\n[2/3] Generating Graco Modes Travel System reviews...")

    print("  Generating 55 English reviews...")
    en_prompt = """
Generate exactly 55 realistic customer reviews for the "Graco Modes Travel System Stroller"
sold on a Middle Eastern baby e-commerce platform.

Make them realistic and varied:
- 30 positive reviews (ratings 4-5): mention ease of folding, compatibility with car seat,
  smooth ride, large storage basket, good for travel
- 15 mixed reviews (ratings 3): mention it's heavy to lift, bulky in small cars,
  assembly takes time, but overall functional
- 10 negative reviews (ratings 1-2): mention wheels breaking, difficult to steer,
  poor customer service for repairs, rust after a few months

Add realistic noise:
- Include 5 reviews from moms comparing it to other brands (Bugaboo, Chicco)
- Include 3 reviews where the mom mentions her baby's age
- Include 4 short reviews (under 15 words)
- 2 reviews should have rating-text mismatch

Return ONLY a JSON array, no markdown, no explanation.
Each object:
{
  "text": "review text here",
  "rating": 4.0,
  "language": "en",
  "reviewer_name": "Name here"
}
"""
    en_raw = call_gemini(en_prompt)
    en_reviews = parse_json_response(en_raw)
    print(f"  Got {len(en_reviews)} English reviews")
    time.sleep(5)

    print("  Generating 15 Arabic reviews...")
    ar_prompt = """
أنشئ بالضبط 15 تقييماً واقعياً من العملاء لمنتج "عربة أطفال غراكو مودز"
على منصة تجارة إلكترونية للأطفال في الشرق الأوسط.

- 9 تقييمات إيجابية (تقييم 4-5): سهولة الطي، مريحة للطفل، مساحة تخزين كبيرة
- 4 تقييمات متوسطة (تقييم 3): ثقيلة نسبياً، لكن جودتها جيدة
- 2 تقييمات سلبية (تقييم 1-2): مشاكل في العجلات أو الصدأ

أعد فقط مصفوفة JSON بدون markdown.
كل كائن:
{
  "text": "نص التقييم",
  "rating": 4.0,
  "language": "ar",
  "reviewer_name": "الاسم هنا"
}
"""
    ar_raw = call_gemini(ar_prompt)
    ar_reviews = parse_json_response(ar_raw)
    print(f"  Got {len(ar_reviews)} Arabic reviews")
    time.sleep(5)

    all_reviews = en_reviews + ar_reviews

    product = {
        "product_name": "Graco Modes Travel System Stroller",
        "product_id": "graco-modes-travel",
        "category": "Strollers",
        "expected_verdict": "mixed",
        "expected_confidence": "high",
        "notes": "Medium-large genuine dataset. Mixed sentiment — good features but heavy and bulky complaints are real.",
        "reviews": all_reviews
    }

    path = OUTPUT_DIR / "graco_stroller.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(product, f, ensure_ascii=False, indent=2)

    print(f"  Saved {len(all_reviews)} reviews to {path}")
    return product


# ---------------------------------------------------------------------------
# Product 3: Generic Brand Baby Bottle (Fake Reviews)
# Small dataset, reviews are suspiciously similar
# Tests: fake review detection, confidence penalty, flagging
# ---------------------------------------------------------------------------

def generate_fake_reviews() -> dict:
    print("\n[3/3] Generating Generic Brand Baby Bottle (fake reviews)...")

    print("  Generating 30 suspiciously similar fake reviews...")
    fake_prompt = """
Generate exactly 30 fake/spam customer reviews for a generic "BabyBliss Premium Bottle 240ml".

These should look like coordinated fake reviews — written by the same person or a bot:
- All ratings are 5 stars
- All reviews are very positive, no complaints at all
- Reviews use very similar sentence structure and vocabulary
- They all mention roughly the same 3-4 talking points:
  "great quality", "my baby loves it", "highly recommend", "best bottle ever"
- Some reviews are nearly word-for-word copies with minor word swaps
- Reviewer names are generic: "Happy Mom", "Satisfied Customer", "MomOf2", etc.
- Length is uniform: all between 20-40 words

This is intentionally unrealistic — we want the fake detector to catch this.

Return ONLY a JSON array, no markdown, no explanation.
Each object:
{
  "text": "review text here",
  "rating": 5.0,
  "language": "en",
  "reviewer_name": "name here"
}
"""
    fake_raw = call_gemini(fake_prompt)
    fake_reviews = parse_json_response(fake_raw)
    print(f"  Got {len(fake_reviews)} fake reviews")
    time.sleep(5)

    product = {
        "product_name": "BabyBliss Premium Baby Bottle 240ml",
        "product_id": "babybliss-premium-240ml",
        "category": "Feeding",
        "expected_verdict": "insufficient_or_flagged",
        "expected_confidence": "low",
        "notes": "FAKE REVIEW TEST CASE. Reviews are intentionally similar. Fake detector should flag this and reduce confidence.",
        "reviews": fake_reviews
    }

    path = OUTPUT_DIR / "fake_reviews.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(product, f, ensure_ascii=False, indent=2)

    print(f"  Saved {len(fake_reviews)} fake reviews to {path}")
    return product


# ---------------------------------------------------------------------------
# Product 4: Edge case — too few reviews
# Tests: INSUFFICIENT refusal
# ---------------------------------------------------------------------------

def generate_sparse_reviews() -> dict:
    print("\n[4/4] Generating sparse review dataset (edge case)...")

    # we write these manually — no need for an API call
    # 3 reviews is deliberately below our minimum of 5
    sparse_reviews = [
        {
            "text": "Just received it, seems good so far. Will update after a few weeks.",
            "rating": 4.0,
            "language": "en",
            "reviewer_name": "New Mom"
        },
        {
            "text": "My baby likes it.",
            "rating": 5.0,
            "language": "en",
            "reviewer_name": "Fatima A."
        },
        {
            "text": "مناسبة للسعر",
            "rating": 3.0,
            "language": "ar",
            "reviewer_name": "أم خالد"
        }
    ]

    product = {
        "product_name": "NovaBaby Silicone Bottle 150ml",
        "product_id": "novababy-silicone-150ml",
        "category": "Feeding",
        "expected_verdict": "refused",
        "expected_confidence": "insufficient",
        "notes": "EDGE CASE: Only 3 reviews. Pipeline must refuse to generate verdict and populate refusal_reason.",
        "reviews": sparse_reviews
    }

    path = OUTPUT_DIR / "sparse_reviews.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(product, f, ensure_ascii=False, indent=2)

    print(f"  Saved {len(sparse_reviews)} reviews to {path}")
    return product


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Moms Verdict — Synthetic Review Generator")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")

    products = []

    products.append(generate_avent_bottle_reviews())
    products.append(generate_graco_stroller_reviews())
    products.append(generate_fake_reviews())
    products.append(generate_sparse_reviews())

    # save a combined index file for easy loading in evals
    index = [
        {
            "product_name": p["product_name"],
            "product_id": p["product_id"],
            "file": f"{p['product_id']}.json".replace(
                p["product_id"],
                p["product_id"].replace("-", "_")
            ),
            "review_count": len(p["reviews"]),
            "expected_verdict": p["expected_verdict"],
            "expected_confidence": p["expected_confidence"],
            "notes": p["notes"]
        }
        for p in products
    ]

    # fix filenames in index to match actual saved files
    filename_map = {
        "avent-natural-260ml": "avent_bottle.json",
        "graco-modes-travel": "graco_stroller.json",
        "babybliss-premium-240ml": "fake_reviews.json",
        "novababy-silicone-150ml": "sparse_reviews.json",
    }
    for item in index:
        item["file"] = filename_map[item["product_id"]]

    index_path = OUTPUT_DIR / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Generation complete.")
    print(f"Products generated: {len(products)}")
    for p in products:
        print(f"  - {p['product_name']}: {len(p['reviews'])} reviews")
    print(f"\nIndex saved to: {index_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
