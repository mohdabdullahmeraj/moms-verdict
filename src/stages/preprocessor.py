"""
src/stages/preprocessor.py

Stage 1: Preprocessing and normalization.

Takes raw review dicts (from JSON files) and produces
clean, validated Review objects ready for the pipeline.

What it does:
- Validates each review against the Review Pydantic schema
- Detects language if not declared (simple heuristic — Arabic Unicode check)
- Clamps ratings to valid range (1.0–5.0)
- Removes reviews that are too short to be meaningful (< 3 chars)
- Flags rating-text mismatches (5 stars but text is negative, etc.)
- Computes language breakdown statistics

What it deliberately does NOT do:
- Translate reviews — we process each language natively downstream
- Remove stopwords or stem text — the embedding model handles this better
- Deduplicate exact copies — that's the fake detector's job
"""

from src.schema import Review


# ---------------------------------------------------------------------------
# Language detection (heuristic, no API needed)
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """
    Simple language detection based on Unicode character ranges.

    Arabic block: U+0600 to U+06FF
    If more than 20% of non-space characters are Arabic, classify as Arabic.
    Otherwise classify as English (default for latin script).

    This is intentionally simple — good enough for EN/AR discrimination.
    For production, you'd use langdetect or a proper language ID model.
    """
    if not text:
        return "en"

    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return "en"

    arabic_chars = sum(1 for c in non_space if '\u0600' <= c <= '\u06FF')
    arabic_ratio = arabic_chars / len(non_space)

    return "ar" if arabic_ratio > 0.20 else "en"


# ---------------------------------------------------------------------------
# Rating-text mismatch detection (heuristic)
# ---------------------------------------------------------------------------

# words that strongly suggest negative sentiment
NEGATIVE_SIGNALS = [
    "terrible", "awful", "horrible", "worst", "broken", "useless",
    "disappointed", "waste", "refund", "return", "leak", "broke",
    "never", "don't buy", "do not buy", "avoid",
    # Arabic negative signals
    "سيء", "مروع", "فظيع", "كسر", "ضائع", "لا أنصح", "مخيب"
]

# words that strongly suggest positive sentiment
POSITIVE_SIGNALS = [
    "love", "amazing", "excellent", "perfect", "best", "fantastic",
    "highly recommend", "great", "wonderful",
    # Arabic positive signals
    "رائع", "ممتاز", "أنصح", "جيد جداً", "أحب"
]


def detect_rating_text_mismatch(text: str, rating: float) -> bool:
    """
    Returns True if the rating and text sentiment appear to contradict.

    Cases we catch:
    - Rating 4-5 but text contains strong negative signals
    - Rating 1-2 but text contains strong positive signals

    This is intentionally conservative — we only flag strong mismatches.
    A 3-star review with mixed language is not a mismatch.
    """
    text_lower = text.lower()

    has_negative = any(signal in text_lower for signal in NEGATIVE_SIGNALS)
    has_positive = any(signal in text_lower for signal in POSITIVE_SIGNALS)

    if rating >= 4.0 and has_negative and not has_positive:
        return True
    if rating <= 2.0 and has_positive and not has_negative:
        return True

    return False


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Cleans raw review data into validated Review objects.

    Usage:
        preprocessor = Preprocessor()
        clean_reviews, stats = preprocessor.run(raw_reviews)
    """

    MIN_TEXT_LENGTH = 3      # reviews shorter than this are dropped
    MAX_TEXT_LENGTH = 2000   # cap extremely long reviews

    def run(self, raw_reviews: list[dict]) -> tuple[list[Review], dict]:
        """
        Process raw review dicts into clean Review objects.

        Args:
            raw_reviews: List of dicts from JSON file.
                         Each dict should have: text, rating, language (optional),
                         reviewer_name (optional)

        Returns:
            tuple of:
            - list[Review]: validated, cleaned Review objects
            - dict: statistics about the preprocessing run
        """
        clean = []
        dropped = []
        mismatches = []

        for i, raw in enumerate(raw_reviews):

            # --- Extract fields with defaults ---
            text = str(raw.get("text", "")).strip()
            rating = float(raw.get("rating", 3.0))
            reviewer_name = raw.get("reviewer_name", None)

            # --- Drop if too short ---
            if len(text) < self.MIN_TEXT_LENGTH:
                dropped.append({
                    "index": i,
                    "reason": f"Too short ({len(text)} chars)",
                    "text": text
                })
                continue

            # --- Truncate if too long ---
            if len(text) > self.MAX_TEXT_LENGTH:
                text = text[:self.MAX_TEXT_LENGTH]

            # --- Clamp rating to valid range ---
            rating = max(1.0, min(5.0, rating))

            # --- Detect language ---
            # use declared language if valid, otherwise detect
            declared_lang = raw.get("language", "")
            if declared_lang in ("en", "ar"):
                language = declared_lang
            else:
                language = detect_language(text)

            # --- Check for rating-text mismatch ---
            mismatch = detect_rating_text_mismatch(text, rating)
            if mismatch:
                mismatches.append({
                    "index": i,
                    "rating": rating,
                    "text_preview": text[:80]
                })

            # --- Build validated Review object ---
            try:
                review = Review(
                    text=text,
                    rating=rating,
                    language=language,
                    reviewer_name=reviewer_name
                )
                clean.append(review)

            except Exception as e:
                # Pydantic validation failed — drop and log
                dropped.append({
                    "index": i,
                    "reason": f"Validation error: {e}",
                    "text": text[:80]
                })

        # --- Compute language breakdown ---
        language_breakdown = {}
        for review in clean:
            language_breakdown[review.language] = (
                language_breakdown.get(review.language, 0) + 1
            )

        stats = {
            "total_input": len(raw_reviews),
            "total_clean": len(clean),
            "total_dropped": len(dropped),
            "total_mismatches": len(mismatches),
            "language_breakdown": language_breakdown,
            "dropped_details": dropped,
            "mismatch_details": mismatches
        }

        print(
            f"  [Preprocessor] {stats['total_input']} in → "
            f"{stats['total_clean']} clean, "
            f"{stats['total_dropped']} dropped, "
            f"{stats['total_mismatches']} rating-text mismatches"
        )

        return clean, stats
