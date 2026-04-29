"""
src/stages/arabic_generator.py

Stage 6: Native Arabic verdict generation.
Updated to use google-genai SDK.
"""

import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

from src.schema import ProConItem, ConfidenceLevel, OverallSentiment
from src.prompts.arabic_prompt import build_arabic_prompt

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

PRO_MODEL = os.environ.get("GEMINI_PRO_MODEL", "gemini-2.0-flash")
MIN_ARABIC_CHARS = 10
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 8


class ArabicGenerator:
    def run(self, product_name, pros, cons, overall_sentiment,
            confidence_level, review_count, language_breakdown):

        prompt = build_arabic_prompt(
            product_name=product_name,
            pros=pros,
            cons=cons,
            overall_sentiment=overall_sentiment,
            confidence_level=confidence_level,
            review_count=review_count,
            language_breakdown=language_breakdown
        )

        for attempt in range(MAX_RETRIES):
            try:
                print(
                    f"  [ArabicGenerator] Generating Arabic verdict "
                    f"(attempt {attempt+1}/{MAX_RETRIES})..."
                )
                response = client.models.generate_content(
                    model=PRO_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.4,
                        max_output_tokens=400
                    )
                )
                candidate = response.text.strip()
                error = self._validate(candidate)
                if error is None:
                    print(
                        f"  [ArabicGenerator] Done "
                        f"({self._count_arabic(candidate)} Arabic chars)"
                    )
                    return candidate
                else:
                    print(f"  [ArabicGenerator] Validation failed: {error}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY_SECONDS)
            except Exception as e:
                print(f"  [ArabicGenerator] API error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS)

        raise ValueError(
            f"ArabicGenerator failed after {MAX_RETRIES} attempts."
        )

    def _validate(self, text):
        if not text or not text.strip():
            return "Empty output"
        if len(text.strip()) < 20:
            return f"Too short: {len(text.strip())} chars"
        if self._count_arabic(text) < MIN_ARABIC_CHARS:
            return f"Not enough Arabic characters: {self._count_arabic(text)}"
        return None

    def _count_arabic(self, text):
        return sum(1 for c in text if '\u0600' <= c <= '\u06FF')
