"""
src/stages/arabic_generator.py

Stage 6: Native Arabic verdict generation using Gemini Pro.

This is one of the most important design decisions in the entire system.

WHY WE DO NOT TRANSLATE:
Translation produces text that reads like translation. Arabic speakers
in the GCC immediately recognise translated copy — the sentence structure
is wrong, the idioms are English idioms rendered in Arabic words, the
register is off. For a platform like Mumzworld whose customers are
largely Arabic-speaking Gulf mothers, this is a trust signal failure.

WHAT WE DO INSTEAD:
We give Gemini the structured facts — the pros, cons, sentiment,
confidence level, and review count — and ask it to write a verdict
in Arabic as a native Arabic-speaking Gulf mother would write it.
The model is not translating verdict_en. It is writing verdict_ar
independently, from the same underlying data.

WHY GEMINI PRO AND NOT FLASH:
Arabic generation requires more nuance than structured extraction.
Flash is faster and cheaper but produces Arabic that occasionally
feels stiff or overly formal. Pro produces more natural Gulf-register
Arabic. We use Pro only for this one call per product — the cost
difference is negligible at prototype scale.

WHY THE PROMPT IS WRITTEN IN ARABIC:
Prompting in Arabic signals the register, formality level, and
cultural framing we want. It also activates the model's Arabic
language capabilities more directly than an English instruction
that says "write in Arabic." This is an empirical observation —
the output quality is measurably better with an Arabic prompt.

WHAT "NATIVE" MEANS HERE:
- Sentence structure follows Arabic grammar, not translated English structure
- Uses Gulf Arabic cultural references where appropriate
  (e.g. "كل أم تبحث عن..." not "every mother is looking for...")
- Formal enough for a product page, warm enough to feel human
- Does not use overly classical (fusha) phrasing that feels stiff
- Does not use colloquialisms that would alienate non-Gulf readers
Simple Modern Standard Arabic with Gulf warmth is the target register.

VALIDATION:
After generation, we validate that the output actually contains
Arabic Unicode characters. This catches the failure mode where
the model returns English in the Arabic field — which happens
occasionally when the model is confused by mixed-language input data.
"""

import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

from src.schema import ProConItem, ConfidenceLevel, OverallSentiment
from src.prompts.arabic_prompt import build_arabic_prompt

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRO_MODEL = os.environ.get("GEMINI_PRO_MODEL", "gemini-1.5-pro")

# slightly higher temperature than extraction —
# we want natural language, not robotic output
ARABIC_GENERATION_CONFIG = genai.GenerationConfig(
    temperature=0.4,
    max_output_tokens=400,    # verdict is 2-3 sentences, 400 tokens is generous
)

# minimum Arabic characters required to pass validation
MIN_ARABIC_CHARS = 10

# retry config
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 8


# ---------------------------------------------------------------------------
# ArabicGenerator
# ---------------------------------------------------------------------------

class ArabicGenerator:
    """
    Generates a native Arabic verdict from structured extraction data.

    Takes the assembled pros, cons, sentiment, and metadata — NOT the
    English verdict — and produces Arabic copy independently.

    Usage:
        generator = ArabicGenerator()
        verdict_ar = generator.run(
            product_name="Philips Avent Bottle",
            pros=pros_list,
            cons=cons_list,
            overall_sentiment=OverallSentiment.POSITIVE,
            confidence_level=ConfidenceLevel.HIGH,
            review_count=133,
            language_breakdown={"en": 98, "ar": 35}
        )
    """

    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name=PRO_MODEL,
            generation_config=ARABIC_GENERATION_CONFIG
        )

    def run(
        self,
        product_name: str,
        pros: list[ProConItem],
        cons: list[ProConItem],
        overall_sentiment: OverallSentiment,
        confidence_level: ConfidenceLevel,
        review_count: int,
        language_breakdown: dict[str, int]
    ) -> str:
        """
        Generate native Arabic verdict.

        Args:
            product_name: Product name (may be in English — that's fine,
                          product names are typically not translated)
            pros: Validated ProConItem list from extraction stage
            cons: Validated ProConItem list from extraction stage
            overall_sentiment: Computed sentiment from pipeline
            confidence_level: Computed confidence level from pipeline
            review_count: Total reviews processed
            language_breakdown: Dict like {"en": 98, "ar": 35}

        Returns:
            Arabic verdict string (2-3 sentences, native Arabic)

        Raises:
            ValueError: If after MAX_RETRIES, no valid Arabic output produced
        """

        # build the Arabic prompt
        prompt = build_arabic_prompt(
            product_name=product_name,
            pros=pros,
            cons=cons,
            overall_sentiment=overall_sentiment,
            confidence_level=confidence_level,
            review_count=review_count,
            language_breakdown=language_breakdown
        )

        # attempt generation with retries
        for attempt in range(MAX_RETRIES):
            try:
                print(
                    f"  [ArabicGenerator] Generating Arabic verdict "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})..."
                )

                response = self.model.generate_content(prompt)
                candidate = response.text.strip()

                # validate Arabic content
                validation_error = self._validate_arabic_output(candidate)

                if validation_error is None:
                    # success
                    print(
                        f"  [ArabicGenerator] Arabic verdict generated "
                        f"({len(candidate)} chars, "
                        f"{self._count_arabic_chars(candidate)} Arabic chars)"
                    )
                    return candidate

                else:
                    print(
                        f"  [ArabicGenerator] Validation failed "
                        f"(attempt {attempt + 1}): {validation_error}"
                    )
                    print(f"  [ArabicGenerator] Output was: {candidate[:100]}...")

                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY_SECONDS)

            except Exception as e:
                print(f"  [ArabicGenerator] API error (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS)

        # all retries exhausted
        raise ValueError(
            f"ArabicGenerator failed after {MAX_RETRIES} attempts. "
            f"Could not produce valid Arabic output for product: {product_name}. "
            f"Check API key, rate limits, and prompt configuration."
        )

    def _validate_arabic_output(self, text: str) -> str | None:
        """
        Validate that the output actually contains Arabic text.

        Returns None if valid, or an error message string if invalid.

        Checks:
        1. Not empty
        2. Contains minimum number of Arabic Unicode characters
        3. Not suspiciously short (less than 20 characters total)
        4. Does not start with common English error patterns
        """
        if not text or not text.strip():
            return "Output is empty"

        if len(text.strip()) < 20:
            return f"Output too short: {len(text.strip())} characters"

        arabic_char_count = self._count_arabic_chars(text)
        if arabic_char_count < MIN_ARABIC_CHARS:
            return (
                f"Insufficient Arabic characters: "
                f"found {arabic_char_count}, need {MIN_ARABIC_CHARS}. "
                f"Output may be in wrong language."
            )

        # check for common failure patterns —
        # model returning English despite Arabic prompt
        english_starts = [
            "i cannot", "i'm unable", "as an ai", "i don't",
            "the product", "this product", "based on"
        ]
        text_lower = text.lower().strip()
        for pattern in english_starts:
            if text_lower.startswith(pattern):
                return f"Output appears to be English (starts with '{pattern}')"

        return None  # valid

    def _count_arabic_chars(self, text: str) -> int:
        """Count characters in the Arabic Unicode block (U+0600 to U+06FF)."""
        return sum(1 for c in text if '\u0600' <= c <= '\u06FF')
