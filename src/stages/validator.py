"""
src/stages/validator.py

Stage 5: Schema validation and confidence computation.

This stage sits between extraction and Arabic generation.
It takes the raw assembled data and:
1. Computes the confidence score mechanically (never the LLM's job)
2. Determines the confidence level tier
3. Deduplicates pros and cons across clusters
4. Enforces the maximum 5 pros / 5 cons limit
5. Determines overall sentiment from cluster signals
6. Prepares the data structure for Arabic generation and final assembly

Design principle:
All logic here is deterministic Python. No LLM calls.
If something can be computed from the data, it is computed —
not inferred by a model.
"""

import os
from dotenv import load_dotenv

from src.schema import (
    ProConItem,
    ConfidenceLevel,
    OverallSentiment,
    FakeReviewFlag
)
from src.stages.extractor import ClusterExtraction

load_dotenv()

# thresholds from environment (with safe defaults)
MIN_REVIEWS_HIGH = int(os.environ.get("MIN_REVIEWS_FOR_HIGH_CONFIDENCE", "50"))
MIN_REVIEWS_MEDIUM = int(os.environ.get("MIN_REVIEWS_FOR_MEDIUM_CONFIDENCE", "15"))
MIN_REVIEWS_VERDICT = int(os.environ.get("MIN_REVIEWS_FOR_VERDICT", "5"))

# confidence penalties
FAKE_FLAG_PENALTY = 0.30
MISMATCH_PENALTY_PER_PERCENT = 0.002  # 0.2% penalty per 1% mismatch rate


class Validator:
    """
    Validates and assembles extraction results into a structured
    pre-verdict data package ready for Arabic generation and
    final MomsVerdict assembly.

    Usage:
        validator = Validator()
        result = validator.run(
            extractions=extractions,
            review_count=133,
            fake_flag=fake_flag,
            language_breakdown={"en": 98, "ar": 35},
            mismatch_rate=0.03
        )
    """

    def run(
        self,
        extractions: list[ClusterExtraction],
        review_count: int,
        fake_flag: FakeReviewFlag,
        language_breakdown: dict[str, int],
        mismatch_rate: float = 0.0
    ) -> dict:
        """
        Validate and assemble extraction results.

        Args:
            extractions: List of ClusterExtraction from Extractor
            review_count: Total reviews processed
            fake_flag: FakeReviewFlag from FakeDetector
            language_breakdown: Dict of language counts
            mismatch_rate: Fraction of reviews with rating-text mismatch

        Returns:
            Dict with assembled, validated data ready for pipeline assembly.
            Keys: pros, cons, overall_sentiment, confidence_score,
                  confidence_level, themes_identified
        """

        print(f"  [Validator] Assembling {len(extractions)} cluster extractions...")

        # --- Compute confidence ---
        confidence_score, confidence_level = self._compute_confidence(
            review_count=review_count,
            fake_flag=fake_flag,
            mismatch_rate=mismatch_rate
        )
        print(
            f"  [Validator] Confidence: {confidence_score} ({confidence_level})"
        )

        # --- Assemble and deduplicate pros/cons ---
        all_pros = self._collect_and_deduplicate(
            [p for e in extractions for p in e.pros]
        )
        all_cons = self._collect_and_deduplicate(
            [c for e in extractions for c in e.cons]
        )

        # sort by mention percentage descending
        all_pros.sort(key=lambda x: x.mention_percentage, reverse=True)
        all_cons.sort(key=lambda x: x.mention_percentage, reverse=True)

        # enforce max 5 each
        final_pros = all_pros[:5]
        final_cons = all_cons[:5]

        print(
            f"  [Validator] Final: {len(final_pros)} pros, "
            f"{len(final_cons)} cons"
        )

        # --- Determine overall sentiment ---
        overall_sentiment = self._determine_sentiment(extractions)
        print(f"  [Validator] Overall sentiment: {overall_sentiment}")

        # --- Collect theme labels ---
        themes = list(dict.fromkeys(
            e.theme_label for e in extractions
        ))  # dict.fromkeys preserves order and deduplicates

        return {
            "pros": final_pros,
            "cons": final_cons,
            "overall_sentiment": overall_sentiment,
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "themes_identified": themes
        }

    def _compute_confidence(
        self,
        review_count: int,
        fake_flag: FakeReviewFlag,
        mismatch_rate: float
    ) -> tuple[float, ConfidenceLevel]:
        """
        Compute confidence score and level mechanically.

        Base score is determined by review volume.
        Penalties are applied for fake flag and rating-text mismatches.
        Final score is clamped to [0.0, 1.0].
        Level is assigned based on final score and review count.

        This is entirely deterministic. The LLM never touches this.
        """

        # base score from volume
        if review_count < MIN_REVIEWS_VERDICT:
            return 0.0, ConfidenceLevel.INSUFFICIENT

        elif review_count < MIN_REVIEWS_MEDIUM:
            base = 0.50
            base_level = ConfidenceLevel.LOW

        elif review_count < MIN_REVIEWS_HIGH:
            base = 0.72
            base_level = ConfidenceLevel.MEDIUM

        else:
            base = 0.90
            base_level = ConfidenceLevel.HIGH

        # apply penalties
        score = base

        if fake_flag.flagged:
            score -= FAKE_FLAG_PENALTY
            print(
                f"  [Validator] Fake flag penalty applied: "
                f"-{FAKE_FLAG_PENALTY} → {score:.3f}"
            )

        if mismatch_rate > 0:
            mismatch_penalty = mismatch_rate * 100 * MISMATCH_PENALTY_PER_PERCENT
            score -= mismatch_penalty
            print(
                f"  [Validator] Mismatch penalty applied: "
                f"-{mismatch_penalty:.3f} → {score:.3f}"
            )

        # clamp to valid range
        score = round(max(0.0, min(1.0, score)), 3)

        # re-evaluate level based on final score
        # (penalties can drop a HIGH review count into MEDIUM territory)
        if score < 0.40:
            final_level = ConfidenceLevel.LOW
        elif score < 0.65:
            final_level = ConfidenceLevel.MEDIUM
        else:
            final_level = base_level

        return score, final_level

    def _collect_and_deduplicate(
        self,
        items: list[ProConItem]
    ) -> list[ProConItem]:
        """
        Deduplicate ProConItems across clusters.

        Two items are considered duplicates if their point text is
        very similar (first 30 characters match after lowercasing).
        When duplicates are found, keep the one with higher mention_count.

        This prevents the same point appearing twice in the final output
        (e.g. "Easy to clean" from cluster 1 and "Easy to clean" from cluster 3).
        """
        seen: dict[str, ProConItem] = {}

        for item in items:
            # use first 30 chars of lowercased point as dedup key
            key = item.point.lower().strip()[:30]

            if key not in seen:
                seen[key] = item
            else:
                # keep the one with higher mention count
                if item.evidence.mention_count > seen[key].evidence.mention_count:
                    seen[key] = item

        return list(seen.values())

    def _determine_sentiment(
        self,
        extractions: list[ClusterExtraction]
    ) -> OverallSentiment:
        """
        Determine overall sentiment from cluster-level sentiment signals.

        Weight each cluster's signal by its review count —
        larger clusters have more influence on the overall sentiment.

        Thresholds:
        - positive: weighted positive fraction > 0.65
        - negative: weighted negative fraction > 0.65
        - mixed: everything else
        """
        if not extractions:
            return OverallSentiment.MIXED

        total_reviews = sum(e.review_count for e in extractions)
        if total_reviews == 0:
            return OverallSentiment.MIXED

        positive_weight = 0.0
        negative_weight = 0.0

        for e in extractions:
            weight = e.review_count / total_reviews
            if e.sentiment_signal == "positive":
                positive_weight += weight
            elif e.sentiment_signal == "negative":
                negative_weight += weight
            # "mixed" adds to neither — reduces both fractions

        if positive_weight > 0.65:
            return OverallSentiment.POSITIVE
        elif negative_weight > 0.65:
            return OverallSentiment.NEGATIVE
        else:
            return OverallSentiment.MIXED
