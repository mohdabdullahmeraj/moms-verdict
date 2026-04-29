"""
src/pipeline.py

The main pipeline orchestrator for Moms Verdict.

This is the single entry point for the entire system.
Call pipeline.run(product_name, raw_reviews) and get back
a fully validated MomsVerdict object.

Architecture overview:
    Stage 1 — Preprocessor:      clean raw reviews → list[Review]
    Stage 2 — FakeDetector:      detect spam → FakeReviewFlag
    Stage 3 — Clusterer:         group by theme → list[ReviewCluster]
    Stage 4 — Extractor:         LLM extraction → list[ClusterExtraction]
    Stage 5 — Validator:         assemble + confidence → validated dict
    Stage 6 — ArabicGenerator:   native Arabic → verdict_ar string
    Stage 7 — Assembly:          build final MomsVerdict + Pydantic validation

Design principles:
- Each stage is independently testable.
- Failures are explicit. No stage swallows exceptions silently.
- The pipeline has two special paths:
    INSUFFICIENT path: fewer than MIN_REVIEWS reviews →
        return MomsVerdict with confidence=INSUFFICIENT immediately,
        skip all LLM calls entirely.
    NORMAL path: enough reviews →
        run all stages, assemble full verdict.
- Embeddings are computed once in FakeDetector and passed to Clusterer.
  This avoids computing the same embeddings twice — an efficiency
  decision worth mentioning in TRADEOFFS.md.
- The pipeline is stateless. Each call to run() is fully independent.
  No state is shared between calls. Safe to run concurrently.

Usage:
    from src.pipeline import MomsVerdictPipeline

    pipeline = MomsVerdictPipeline()
    verdict = pipeline.run(
        product_name="Philips Avent Natural Baby Bottle 260ml",
        raw_reviews=[
            {"text": "Love this bottle!", "rating": 5.0, "language": "en"},
            ...
        ]
    )
    print(verdict.model_dump_json(indent=2))
"""

import os
import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.schema import (
    Review,
    MomsVerdict,
    ConfidenceLevel,
    OverallSentiment,
    FakeReviewFlag
)
from src.stages.preprocessor import Preprocessor
from src.stages.fake_detector import FakeReviewDetector
from src.stages.clusterer import Clusterer
from src.stages.extractor import Extractor
from src.stages.validator import Validator
from src.stages.arabic_generator import ArabicGenerator

load_dotenv()

console = Console()

MIN_REVIEWS_FOR_VERDICT = int(
    os.environ.get("MIN_REVIEWS_FOR_VERDICT", "5")
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MomsVerdictPipeline:
    """
    Orchestrates all stages of the Moms Verdict system.

    Instantiate once and reuse — the embedding model is loaded
    lazily on first use and cached inside FakeDetector and Clusterer.

    Example:
        pipeline = MomsVerdictPipeline()

        # normal run
        verdict = pipeline.run("Philips Avent Bottle", reviews)

        # run from a JSON file
        verdict = pipeline.run_from_file("data/sample_products/avent_bottle.json")
    """

    def __init__(self):
        # instantiate all stages
        # embedding model is lazy-loaded on first use inside these classes
        self.preprocessor = Preprocessor()
        self.fake_detector = FakeReviewDetector()
        self.clusterer = Clusterer()
        self.extractor = Extractor()
        self.validator = Validator()
        self.arabic_generator = ArabicGenerator()

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def run(
        self,
        product_name: str,
        raw_reviews: list[dict]
    ) -> MomsVerdict:
        """
        Run the full pipeline for one product.

        Args:
            product_name: Name of the product (used in prompts and output)
            raw_reviews: List of raw review dicts. Each dict should have:
                         - text (str, required)
                         - rating (float, required)
                         - language (str, optional — "en" or "ar")
                         - reviewer_name (str, optional)

        Returns:
            MomsVerdict: Fully validated output object.
                         Always returns — never raises on business logic failures.
                         Raises only on unrecoverable system errors
                         (bad API key, network down, etc.)
        """

        console.print(Panel(
            f"[bold]Moms Verdict Pipeline[/bold]\n"
            f"Product: [cyan]{product_name}[/cyan]\n"
            f"Input reviews: [yellow]{len(raw_reviews)}[/yellow]",
            title="Starting",
            border_style="blue"
        ))

        # ===================================================================
        # STAGE 1: Preprocessing
        # ===================================================================
        console.print("\n[bold blue]Stage 1/6[/bold blue] — Preprocessing...")

        clean_reviews, preprocess_stats = self.preprocessor.run(raw_reviews)

        language_breakdown = preprocess_stats["language_breakdown"]
        mismatch_rate = (
            preprocess_stats["total_mismatches"] /
            max(preprocess_stats["total_clean"], 1)
        )

        console.print(
            f"  ✓ {len(clean_reviews)} clean reviews "
            f"({preprocess_stats['total_dropped']} dropped, "
            f"{preprocess_stats['total_mismatches']} rating-text mismatches)"
        )

        # ===================================================================
        # GATE: INSUFFICIENT CHECK
        # Return immediately if not enough reviews for a reliable verdict.
        # Skip all LLM calls entirely — saves cost and time.
        # ===================================================================
        if len(clean_reviews) < MIN_REVIEWS_FOR_VERDICT:
            console.print(
                f"\n[bold yellow]⚠ INSUFFICIENT DATA[/bold yellow] — "
                f"only {len(clean_reviews)} reviews after preprocessing. "
                f"Minimum required: {MIN_REVIEWS_FOR_VERDICT}."
            )
            return self._build_insufficient_verdict(
                product_name=product_name,
                review_count=len(clean_reviews),
                language_breakdown=language_breakdown
            )

        # ===================================================================
        # STAGE 2: Fake Review Detection
        # ===================================================================
        console.print("\n[bold blue]Stage 2/6[/bold blue] — Fake review detection...")

        fake_flag = self.fake_detector.run(clean_reviews)

        if fake_flag.flagged:
            console.print(
                f"  ⚠ [yellow]Fake reviews detected[/yellow] — "
                f"similarity score: {fake_flag.average_similarity_score:.3f}. "
                f"Confidence will be reduced."
            )
        else:
            console.print(
                f"  ✓ Reviews appear genuine "
                f"(similarity score: {fake_flag.average_similarity_score:.3f})"
            )

        # retrieve embeddings computed inside fake_detector for reuse
        # this avoids recomputing embeddings in the clusterer
        cached_embeddings = self._get_cached_embeddings(clean_reviews)

        # ===================================================================
        # STAGE 3: Theme Clustering
        # ===================================================================
        console.print("\n[bold blue]Stage 3/6[/bold blue] — Theme clustering...")

        clusters = self.clusterer.run(
            reviews=clean_reviews,
            embeddings=cached_embeddings
        )

        console.print(f"  ✓ {len(clusters)} themes identified:")
        for cluster in clusters:
            console.print(
                f"    • '{cluster.theme_label}' "
                f"({len(cluster.reviews)} reviews)"
            )

        # ===================================================================
        # STAGE 4: LLM Extraction (one call per cluster)
        # ===================================================================
        console.print(
            f"\n[bold blue]Stage 4/6[/bold blue] — "
            f"Extracting structured data from {len(clusters)} clusters..."
        )

        extractions = self.extractor.run(
            clusters=clusters,
            product_name=product_name,
            total_review_count=len(clean_reviews)
        )

        if not extractions:
            # all LLM calls failed — this is a system error, not a data error
            raise RuntimeError(
                f"Extraction failed for all {len(clusters)} clusters. "
                f"Check API key, rate limits, and network connectivity."
            )

        console.print(
            f"  ✓ Extracted from {len(extractions)}/{len(clusters)} clusters"
        )

        # ===================================================================
        # STAGE 5: Validation and Assembly
        # ===================================================================
        console.print("\n[bold blue]Stage 5/6[/bold blue] — Validating and assembling...")

        validated = self.validator.run(
            extractions=extractions,
            review_count=len(clean_reviews),
            fake_flag=fake_flag,
            language_breakdown=language_breakdown,
            mismatch_rate=mismatch_rate
        )

        console.print(
            f"  ✓ {len(validated['pros'])} pros, "
            f"{len(validated['cons'])} cons, "
            f"sentiment: {validated['overall_sentiment']}, "
            f"confidence: {validated['confidence_score']} "
            f"({validated['confidence_level']})"
        )

        # ===================================================================
        # GATE: Skip Arabic generation if confidence is INSUFFICIENT
        # (can happen after penalties even if we passed the review count gate)
        # ===================================================================
        if validated["confidence_level"] == ConfidenceLevel.INSUFFICIENT:
            console.print(
                "\n[bold yellow]⚠ Confidence dropped to INSUFFICIENT "
                "after penalties[/bold yellow] — skipping Arabic generation."
            )
            return self._build_insufficient_verdict(
                product_name=product_name,
                review_count=len(clean_reviews),
                language_breakdown=language_breakdown,
                fake_flag=fake_flag,
                reason=(
                    "Confidence score dropped below minimum threshold "
                    "after fake review penalties were applied."
                )
            )

        # ===================================================================
        # STAGE 6: Native Arabic Generation
        # ===================================================================
        console.print("\n[bold blue]Stage 6/6[/bold blue] — Generating Arabic verdict...")

        verdict_ar = self.arabic_generator.run(
            product_name=product_name,
            pros=validated["pros"],
            cons=validated["cons"],
            overall_sentiment=validated["overall_sentiment"],
            confidence_level=validated["confidence_level"],
            review_count=len(clean_reviews),
            language_breakdown=language_breakdown
        )

        console.print(f"  ✓ Arabic verdict generated ({len(verdict_ar)} chars)")

        # ===================================================================
        # STAGE 7: Build English verdict and assemble final MomsVerdict
        # ===================================================================
        console.print("\n[bold blue]Assembling[/bold blue] — Building final verdict...")

        verdict_en = self._build_english_verdict(
            pros=validated["pros"],
            cons=validated["cons"],
            overall_sentiment=validated["overall_sentiment"],
            confidence_level=validated["confidence_level"],
            review_count=len(clean_reviews),
            fake_flagged=fake_flag.flagged
        )

        # assemble and validate final object
        # Pydantic validation runs automatically on construction
        # if this raises, it's a schema violation — a real bug
        verdict = MomsVerdict(
            product_name=product_name,
            verdict_en=verdict_en,
            verdict_ar=verdict_ar,
            pros=validated["pros"],
            cons=validated["cons"],
            overall_sentiment=validated["overall_sentiment"],
            confidence_level=validated["confidence_level"],
            confidence_score=validated["confidence_score"],
            review_count=len(clean_reviews),
            language_breakdown=language_breakdown,
            fake_review_flag=fake_flag,
            themes_identified=validated["themes_identified"],
            refusal_reason=None
        )

        console.print(Panel(
            f"[bold green]✓ Verdict complete[/bold green]\n"
            f"Confidence: [cyan]{verdict.confidence_score}[/cyan] "
            f"({verdict.confidence_level})\n"
            f"Pros: {len(verdict.pros)} | "
            f"Cons: {len(verdict.cons)} | "
            f"Themes: {len(verdict.themes_identified)}\n"
            f"Fake flag: "
            f"{'[red]YES[/red]' if verdict.fake_review_flag.flagged else '[green]NO[/green]'}",
            title="Done",
            border_style="green"
        ))

        return verdict

    # -----------------------------------------------------------------------
    # Convenience loader
    # -----------------------------------------------------------------------

    def run_from_file(self, json_path: str) -> MomsVerdict:
        """
        Load a product JSON file and run the pipeline.

        The JSON file should have the structure produced by generate_reviews.py:
        {
            "product_name": "...",
            "reviews": [{"text": ..., "rating": ..., "language": ...}, ...]
        }

        Args:
            json_path: Path to the product JSON file

        Returns:
            MomsVerdict
        """
        import json
        from pathlib import Path

        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Product file not found: {json_path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        product_name = data.get("product_name", "Unknown Product")
        raw_reviews = data.get("reviews", [])

        console.print(
            f"\n[dim]Loaded {len(raw_reviews)} reviews from {path.name}[/dim]"
        )

        return self.run(product_name, raw_reviews)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_cached_embeddings(
        self,
        reviews: list[Review]
    ) -> np.ndarray | None:
        """
        Retrieve embeddings computed during fake detection.

        The FakeDetector computes embeddings internally but doesn't
        expose them directly. We recompute here using the same model
        that's already loaded in the detector.

        In a production system you'd refactor FakeDetector to return
        embeddings alongside the flag. For prototype purposes,
        recomputing is acceptable — the model is already in memory
        so it's fast.

        We document this as a known inefficiency in TRADEOFFS.md.
        """
        try:
            texts = [r.text for r in reviews]
            embeddings = self.fake_detector.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            console.print(
                f"  [dim yellow]Could not retrieve cached embeddings: {e}. "
                f"Clusterer will recompute.[/dim yellow]"
            )
            return None

    def _build_english_verdict(
        self,
        pros: list,
        cons: list,
        overall_sentiment: OverallSentiment,
        confidence_level: ConfidenceLevel,
        review_count: int,
        fake_flagged: bool
    ) -> str:
        """
        Build the English verdict programmatically from structured data.

        We do NOT call the LLM for the English verdict.
        We construct it from the top pro, top con, sentiment,
        and confidence — deterministically.

        Why not use the LLM for English too?
        - The Arabic verdict needs LLM because natural language generation
          in Arabic is genuinely hard and culturally nuanced.
        - The English verdict can be constructed reliably from a template
          using the structured data we already have.
        - This saves one LLM call per product and makes the English
          output fully deterministic and testable.
        - We document this asymmetry in TRADEOFFS.md.

        The template is intentionally simple and honest —
        it reads like a trusted friend, not marketing copy.
        """

        parts = []

        # opening: sentiment + top pro
        if pros:
            top_pro = pros[0]
            pct = int(top_pro.mention_percentage)

            if overall_sentiment == OverallSentiment.POSITIVE:
                opener = (
                    f"Most moms are happy with this product — "
                    f"{pct}% specifically highlight {top_pro.evidence.claim}."
                )
            elif overall_sentiment == OverallSentiment.NEGATIVE:
                opener = (
                    f"Most moms have concerns about this product, "
                    f"though {pct}% do appreciate {top_pro.evidence.claim}."
                )
            else:
                opener = (
                    f"Moms have mixed feelings about this product — "
                    f"{pct}% highlight {top_pro.evidence.claim} as a plus."
                )
        else:
            opener = (
                f"Moms have {'positive' if overall_sentiment == OverallSentiment.POSITIVE else 'mixed'} "
                f"feelings about this product overall."
            )

        parts.append(opener)

        # middle: top con if exists
        if cons:
            top_con = cons[0]
            con_pct = int(top_con.mention_percentage)
            parts.append(
                f"The main concern raised by {con_pct}% of reviewers "
                f"is {top_con.evidence.claim}."
            )

        # closing: confidence note
        if confidence_level == ConfidenceLevel.LOW:
            parts.append(
                f"Note: this verdict is based on only {review_count} reviews "
                f"and may not be fully representative."
            )
        elif fake_flagged:
            parts.append(
                f"Note: some reviews showed unusual similarity — "
                f"treat this verdict with some caution."
            )

        return " ".join(parts)

    def _build_insufficient_verdict(
        self,
        product_name: str,
        review_count: int,
        language_breakdown: dict,
        fake_flag: FakeReviewFlag | None = None,
        reason: str | None = None
    ) -> MomsVerdict:
        """
        Build a MomsVerdict for the INSUFFICIENT case.

        Called when:
        - Fewer than MIN_REVIEWS reviews after preprocessing
        - Confidence drops to INSUFFICIENT after penalties

        No LLM calls are made in this path.
        verdict_en and verdict_ar are empty strings (required by schema).
        refusal_reason explains why.
        """

        if fake_flag is None:
            fake_flag = FakeReviewFlag(
                flagged=False,
                average_similarity_score=0.0,
                reason="Fake detection skipped — insufficient reviews."
            )

        refusal_reason = reason or (
            f"Only {review_count} reviews available after preprocessing. "
            f"A minimum of {MIN_REVIEWS_FOR_VERDICT} reviews is required "
            f"to generate a reliable verdict for '{product_name}'. "
            f"Please check back when more reviews have been submitted."
        )

        return MomsVerdict(
            product_name=product_name,
            verdict_en="",
            verdict_ar="",
            pros=[],
            cons=[],
            overall_sentiment=OverallSentiment.MIXED,
            confidence_level=ConfidenceLevel.INSUFFICIENT,
            confidence_score=0.0,
            review_count=review_count,
            language_breakdown=language_breakdown or {},
            fake_review_flag=fake_flag,
            themes_identified=[],
            refusal_reason=refusal_reason
        )
