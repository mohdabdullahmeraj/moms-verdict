"""
src/stages/arabic_generator.py

Stage 6: Native Arabic verdict generation via OpenRouter.
Uses Llama 3.3 70B — strong multilingual model, free on OpenRouter.
"""

import os
import time
import openai
from dotenv import load_dotenv

from src.schema import ProConItem, ConfidenceLevel, OverallSentiment
from src.prompts.arabic_prompt import build_arabic_prompt

load_dotenv()

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    default_headers={
        "HTTP-Referer": "https://github.com/moms-verdict",
        "X-Title": "Moms Verdict"
    }
)

ARABIC_MODEL = os.environ.get("ARABIC_MODEL", "openai/gpt-oss-20b:free")
MIN_ARABIC_CHARS = 10
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10


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

                response = client.chat.completions.create(
                    model=ARABIC_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "أنتِ محررة محتوى عربية متخصصة في منتجات الأمومة. "
                                "أجيبي بالعربية فقط. لا تضيفي أي تنسيق أو شرح."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.4,
                    max_tokens=400
                )

                candidate = response.choices[0].message.content.strip()
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
            f"ArabicGenerator failed after {MAX_RETRIES} attempts "
            f"for product: {product_name}"
        )

    def _validate(self, text):
        if not text or not text.strip():
            return "Empty output"
        if len(text.strip()) < 20:
            return f"Too short: {len(text.strip())} chars"
        if self._count_arabic(text) < MIN_ARABIC_CHARS:
            return f"Not enough Arabic chars: {self._count_arabic(text)}"
        return None

    def _count_arabic(self, text):
        return sum(1 for c in text if '\u0600' <= c <= '\u06FF')
