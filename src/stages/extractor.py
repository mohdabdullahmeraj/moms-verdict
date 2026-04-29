"""
src/stages/extractor.py

Stage 4: Structured extraction via Gemini.

Takes themed review clusters from the Clusterer and produces
structured extraction results: pros, cons, sentiment, and an
English verdict — all grounded in actual review quotes.

Design decisions:
- One LLM call per cluster, not one call for all reviews at once.
  This produces more specific, better-grounded output because the model
  focuses on one theme at a time.
- We pass the full schema as part of the prompt so the model knows
  exactly what JSON shape to produce.
- We use JSON mode (response_mime_type) to enforce structured output.
  This eliminates markdown fence stripping and reduces parse failures.
- Extraction produces an intermediate result (ExtractionResult) that
  is separate from the final MomsVerdict. The pipeline assembles
  the final verdict from multiple extraction results.
- Temperature is set to 0.2 — low randomness for factual extraction.
  We want consistency, not creativity.

Grounding enforcement:
- The prompt explicitly instructs: every claim needs a quote.
- If the model cannot find a quote, it must not include the claim.
- We validate this with Pydantic after receiving the response.
- Any ProConItem that slips through without a quote fails validation.
"""

import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv

from src.schema import Review, ProConItem, SupportingEvidence, OverallSentiment
from src.stages.clusterer import ReviewCluster
from src.prompts.extraction_prompt import build_extraction_prompt

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLASH_MODEL = os.environ.get("GEMINI_FLASH_MODEL", "gemini-1.5-flash")

# max reviews per cluster to send to LLM
# prevents token overflow on very large clusters
MAX_REVIEWS_PER_CLUSTER = 40

# delay between cluster LLM calls to respect rate limits
INTER_CALL_DELAY_SECONDS = 3

# generation config — low temperature for factual extraction
GENERATION_CONFIG = genai.GenerationConfig(
    temperature=0.2,
    max_output_tokens=2000,
    response_mime_type="application/json"  # enforces JSON output
)


# ---------------------------------------------------------------------------
# Intermediate data structures
# ---------------------------------------------------------------------------

class ClusterExtraction:
    """
    The structured output from processing one cluster.
    Multiple ClusterExtractions are merged into the final MomsVerdict.
    """
    def __init__(
        self,
        cluster_id: int,
        theme_label: str,
        pros: list[ProConItem],
        cons: list[ProConItem],
        sentiment_signal: str,   # "positive", "mixed", or "negative"
        review_count: int
    ):
        self.cluster_id = cluster_id
        self.theme_label = theme_label
        self.pros = pros
        self.cons = cons
        self.sentiment_signal = sentiment_signal
        self.review_count = review_count


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class Extractor:
    """
    Calls Gemini to extract structured pros, cons, and sentiment
    from each review cluster.

    Usage:
        extractor = Extractor()
        extractions = extractor.run(clusters, product_name, total_review_count)
    """

    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name=FLASH_MODEL,
            generation_config=GENERATION_CONFIG
        )

    def run(
        self,
        clusters: list[ReviewCluster],
        product_name: str,
        total_review_count: int
    ) -> list[ClusterExtraction]:
        """
        Process each cluster through Gemini and return extraction results.

        Args:
            clusters: Themed review clusters from Clusterer
            product_name: Product name (for prompt context)
            total_review_count: Total reviews across all clusters
                                 (used to compute mention_percentage)

        Returns:
            List of ClusterExtraction objects, one per cluster
        """
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

            # rate limit pause between calls
            if i < len(clusters) - 1:
                time.sleep(INTER_CALL_DELAY_SECONDS)

        print(
            f"  [Extractor] Completed. "
            f"{len(extractions)}/{len(clusters)} clusters extracted successfully."
        )
        return extractions

    def _process_cluster(
        self,
        cluster: ReviewCluster,
        product_name: str,
        total_review_count: int
    ) -> ClusterExtraction | None:
        """
        Run one LLM call for one cluster.

        Returns ClusterExtraction on success, None on failure.
        Failures are logged but don't crash the pipeline —
        the final verdict is assembled from whatever extractions succeed.
        """

        # sample reviews if cluster is very large
        reviews_to_send = cluster.reviews
        if len(cluster.reviews) > MAX_REVIEWS_PER_CLUSTER:
            # take a representative sample — first 40 after shuffling
            import random
            rng = random.Random(42)  # deterministic sample
            reviews_to_send = rng.sample(cluster.reviews, MAX_REVIEWS_PER_CLUSTER)
            print(
                f"  [Extractor] Cluster has {len(cluster.reviews)} reviews — "
                f"sampling {MAX_REVIEWS_PER_CLUSTER} for LLM call."
            )

        # format reviews for the prompt
        reviews_text = self._format_reviews_for_prompt(reviews_to_send)

        # build the prompt
        prompt = build_extraction_prompt(
            product_name=product_name,
            theme_label=cluster.theme_label,
            reviews_text=reviews_text,
            review_count=len(cluster.reviews),
            total_review_count=total_review_count
        )

        # call Gemini
        try:
            response = self.model.generate_content(prompt)
            raw_json = response.text
        except Exception as e:
            print(f"  [Extractor] API error on cluster {cluster.cluster_id}: {e}")
            return None

        # parse and validate response
        try:
            return self._parse_response(
                raw_json=raw_json,
                cluster=cluster,
                total_review_count=total_review_count
            )
        except Exception as e:
            print(
                f"  [Extractor] Parse/validation error on "
                f"cluster {cluster.cluster_id}: {e}"
            )
            print(f"  [Extractor] Raw response: {raw_json[:200]}...")
            return None

    def _format_reviews_for_prompt(self, reviews: list[Review]) -> str:
        """
        Format review objects into a numbered list for the prompt.

        Format:
            [1] (5★) Sarah M.: "love the wide neck, so easy to fill..."
            [2] (2★) Anonymous: "leaked all over my bag after one week..."

        The number in brackets lets the model reference specific reviews
        when it provides quotes. The star rating gives sentiment context.
        """
        lines = []
        for i, review in enumerate(reviews, start=1):
            stars = "★" * int(review.rating) + "☆" * (5 - int(review.rating))
            name = review.reviewer_name or "Anonymous"
            lines.append(f'[{i}] ({stars}) {name}: "{review.text}"')
        return "\n".join(lines)

    def _parse_response(
        self,
        raw_json: str,
        cluster: ReviewCluster,
        total_review_count: int
    ) -> ClusterExtraction:
        """
        Parse and validate the LLM's JSON response.

        Expected JSON shape from the LLM:
        {
          "pros": [
            {
              "point": "Easy to clean",
              "mention_count": 12,
              "representative_quote": "washes perfectly in the dishwasher",
              "quote_language": "en"
            }
          ],
          "cons": [...],
          "sentiment": "positive"
        }
        """
        data = json.loads(raw_json)

        # parse pros
        pros = []
        for item in data.get("pros", []):
            evidence = SupportingEvidence(
                claim=item.get("point", ""),
                mention_count=int(item.get("mention_count", 1)),
                representative_quote=item.get("representative_quote", ""),
                quote_language=item.get("quote_language", "en")
            )
            mention_pct = round(
                (evidence.mention_count / total_review_count) * 100, 1
            )
            pros.append(ProConItem(
                point=item.get("point", ""),
                evidence=evidence,
                mention_percentage=min(mention_pct, 100.0)
            ))

        # parse cons
        cons = []
        for item in data.get("cons", []):
            evidence = SupportingEvidence(
                claim=item.get("point", ""),
                mention_count=int(item.get("mention_count", 1)),
                representative_quote=item.get("representative_quote", ""),
                quote_language=item.get("quote_language", "en")
            )
            mention_pct = round(
                (evidence.mention_count / total_review_count) * 100, 1
            )
            cons.append(ProConItem(
                point=item.get("point", ""),
                evidence=evidence,
                mention_percentage=min(mention_pct, 100.0)
            ))

        sentiment = data.get("sentiment", "mixed")

        return ClusterExtraction(
            cluster_id=cluster.cluster_id,
            theme_label=cluster.theme_label,
            pros=pros,
            cons=cons,
            sentiment_signal=sentiment,
            review_count=len(cluster.reviews)
        )
