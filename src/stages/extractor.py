"""
src/stages/extractor.py

Stage 4: Structured extraction via Gemini.
Updated to use google-genai SDK (replaces deprecated google.generativeai).
"""

import os
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

from src.schema import Review, ProConItem, SupportingEvidence, OverallSentiment
from src.stages.clusterer import ReviewCluster
from src.prompts.extraction_prompt import build_extraction_prompt

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

FLASH_MODEL = os.environ.get("GEMINI_FLASH_MODEL", "gemini-2.0-flash")
MAX_REVIEWS_PER_CLUSTER = 40
INTER_CALL_DELAY_SECONDS = 4


class ClusterExtraction:
    def __init__(self, cluster_id, theme_label, pros, cons,
                 sentiment_signal, review_count):
        self.cluster_id = cluster_id
        self.theme_label = theme_label
        self.pros = pros
        self.cons = cons
        self.sentiment_signal = sentiment_signal
        self.review_count = review_count


class Extractor:
    def run(self, clusters, product_name, total_review_count):
        extractions = []
        for i, cluster in enumerate(clusters):
            print(
                f"  [Extractor] Processing cluster {i+1}/{len(clusters)}: "
                f"'{cluster.theme_label}' ({len(cluster.reviews)} reviews)"
            )
            extraction = self._process_cluster(
                cluster=cluster,
                product_name=product_name,
                total_review_count=total_review_count
            )
            if extraction is not None:
                extractions.append(extraction)
            if i < len(clusters) - 1:
                time.sleep(INTER_CALL_DELAY_SECONDS)

        print(
            f"  [Extractor] Completed. "
            f"{len(extractions)}/{len(clusters)} clusters extracted."
        )
        return extractions

    def _process_cluster(self, cluster, product_name, total_review_count):
        reviews_to_send = cluster.reviews
        if len(cluster.reviews) > MAX_REVIEWS_PER_CLUSTER:
            import random
            rng = random.Random(42)
            reviews_to_send = rng.sample(cluster.reviews, MAX_REVIEWS_PER_CLUSTER)

        reviews_text = self._format_reviews_for_prompt(reviews_to_send)
        prompt = build_extraction_prompt(
            product_name=product_name,
            theme_label=cluster.theme_label,
            reviews_text=reviews_text,
            review_count=len(cluster.reviews),
            total_review_count=total_review_count
        )

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=FLASH_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=2000,
                        response_mime_type="application/json"
                    )
                )
                raw_json = response.text
                return self._parse_response(raw_json, cluster, total_review_count)
            except Exception as e:
                print(f"  [Extractor] Error attempt {attempt+1}: {e}")
                if attempt < 2:
                    time.sleep(10)
        return None

    def _format_reviews_for_prompt(self, reviews):
        lines = []
        for i, review in enumerate(reviews, start=1):
            stars = "★" * int(review.rating) + "☆" * (5 - int(review.rating))
            name = review.reviewer_name or "Anonymous"
            lines.append(f'[{i}] ({stars}) {name}: "{review.text}"')
        return "\n".join(lines)

    def _parse_response(self, raw_json, cluster, total_review_count):
        # strip markdown fences if present
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]

        data = json.loads(cleaned.strip())

        pros = []
        for item in data.get("pros", []):
            if not item.get("representative_quote", "").strip():
                continue
            evidence = SupportingEvidence(
                claim=item.get("point", "")[:100],
                mention_count=max(1, int(item.get("mention_count", 1))),
                representative_quote=item.get("representative_quote", ""),
                quote_language=item.get("quote_language", "en")
            )
            mention_pct = round(
                (evidence.mention_count / max(total_review_count, 1)) * 100, 1
            )
            pros.append(ProConItem(
                point=item.get("point", "")[:150],
                evidence=evidence,
                mention_percentage=min(mention_pct, 100.0)
            ))

        cons = []
        for item in data.get("cons", []):
            if not item.get("representative_quote", "").strip():
                continue
            evidence = SupportingEvidence(
                claim=item.get("point", "")[:100],
                mention_count=max(1, int(item.get("mention_count", 1))),
                representative_quote=item.get("representative_quote", ""),
                quote_language=item.get("quote_language", "en")
            )
            mention_pct = round(
                (evidence.mention_count / max(total_review_count, 1)) * 100, 1
            )
            cons.append(ProConItem(
                point=item.get("point", "")[:150],
                evidence=evidence,
                mention_percentage=min(mention_pct, 100.0)
            ))

        return ClusterExtraction(
            cluster_id=cluster.cluster_id,
            theme_label=cluster.theme_label,
            pros=pros,
            cons=cons,
            sentiment_signal=data.get("sentiment", "mixed"),
            review_count=len(cluster.reviews)
        )
