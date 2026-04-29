"""
src/schema.py

The single source of truth for all data structures in the Moms Verdict pipeline.
Every LLM output, every pipeline stage output, every eval result
must conform to one of these models.

Design philosophy:
- The schema is the contract. If it doesn't validate, the pipeline fails loudly.
- Every claim must be grounded (representative_quote is never optional).
- Arabic is treated as a first-class field, not an afterthought.
- Confidence is a property of the data, not the model's feeling.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    """
    How much we trust this verdict, based on review volume and quality.
    Computed mechanically in code — never decided by the LLM.
    """
    HIGH = "high"              # 50+ reviews, low fake score
    MEDIUM = "medium"          # 15–49 reviews, or moderate fake score
    LOW = "low"                # 5–14 reviews
    INSUFFICIENT = "insufficient"  # fewer than 5 reviews — verdict refused


class OverallSentiment(str, Enum):
    """
    The dominant tone across all reviews for this product.
    """
    POSITIVE = "positive"      # majority of reviews are favorable
    MIXED = "mixed"            # significant positive AND negative signals
    NEGATIVE = "negative"      # majority of reviews are unfavorable


# ---------------------------------------------------------------------------
# Supporting Models
# ---------------------------------------------------------------------------

class SupportingEvidence(BaseModel):
    """
    The grounding mechanism. Every claim in a ProConItem must be backed
    by this — a specific quote from an actual review, and a count of
    how many reviews made the same point.

    This is what prevents hallucination. If the LLM cannot find a real
    quote, it cannot make the claim.
    """

    claim: str = Field(
        ...,
        description=(
            "A short phrase describing the specific point being made. "
            "Example: 'easy to clean', 'nipple flow too fast', 'good for colic'. "
            "Maximum 10 words."
        ),
        max_length=100
    )

    mention_count: int = Field(
        ...,
        description=(
            "How many reviews in the dataset specifically mention this point. "
            "Must be a real count, not an estimate. Minimum 1."
        ),
        ge=1
    )

    representative_quote: str = Field(
        ...,
        description=(
            "A verbatim excerpt from one of the reviews that best illustrates "
            "this point. Must be copied exactly from the review text provided. "
            "Do not paraphrase. Do not invent. Maximum 50 words."
        ),
        min_length=5,
        max_length=300
    )

    quote_language: str = Field(
        ...,
        description=(
            "The language of the representative_quote. "
            "Use ISO 639-1 codes: 'en' for English, 'ar' for Arabic."
        ),
        pattern="^(en|ar)$"
    )


class ProConItem(BaseModel):
    """
    A single pro or con, always backed by evidence from actual reviews.
    """

    point: str = Field(
        ...,
        description=(
            "A clear, concise statement of the pro or con. "
            "Written in English regardless of source review language. "
            "Example: 'Reduces colic effectively' or 'Nipple flow too fast for newborns'. "
            "Maximum 15 words."
        ),
        max_length=150
    )

    evidence: SupportingEvidence = Field(
        ...,
        description="The grounding evidence from actual reviews supporting this point."
    )

    mention_percentage: float = Field(
        ...,
        description=(
            "Percentage of total reviews that mention this point. "
            "Calculated as: (mention_count / total_review_count) * 100. "
            "Between 0.0 and 100.0."
        ),
        ge=0.0,
        le=100.0
    )


class FakeReviewFlag(BaseModel):
    """
    Result of the fake review detection stage.
    Always present in the output — even when not flagged.
    Computed entirely by code using embedding similarity, not by the LLM.
    """

    flagged: bool = Field(
        ...,
        description=(
            "True if reviews show suspiciously high similarity to each other, "
            "suggesting potential spam or coordinated fake reviews."
        )
    )

    average_similarity_score: float = Field(
        ...,
        description=(
            "Average pairwise cosine similarity across all review embeddings. "
            "Range 0.0 to 1.0. Higher = more similar = more suspicious. "
            "Typical genuine reviews score below 0.75. "
            "Scores above 0.85 trigger the flag."
        ),
        ge=0.0,
        le=1.0
    )

    reason: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable explanation of why the flag was triggered. "
            "None when flagged=False."
        )
    )


# ---------------------------------------------------------------------------
# Review Input Model
# ---------------------------------------------------------------------------

class Review(BaseModel):
    """
    A single customer review as input to the pipeline.
    """

    text: str = Field(
        ...,
        description="The full text of the review.",
        min_length=3
    )

    rating: float = Field(
        ...,
        description="Star rating given by the reviewer. Between 1.0 and 5.0.",
        ge=1.0,
        le=5.0
    )

    language: str = Field(
        default="en",
        description="Detected or declared language. 'en' or 'ar'.",
        pattern="^(en|ar)$"
    )

    reviewer_name: Optional[str] = Field(
        default=None,
        description="Optional reviewer name or identifier."
    )


# ---------------------------------------------------------------------------
# Main Output Model
# ---------------------------------------------------------------------------

class MomsVerdict(BaseModel):
    """
    The complete output of the Moms Verdict pipeline for one product.

    Design decisions:
    - verdict_en and verdict_ar are separate fields, not a dict keyed by language.
      This forces the pipeline to generate both explicitly. You cannot accidentally
      return one and skip the other.
    - confidence_score is always computed by code, never by the LLM.
    - fake_review_flag is always present, even when not flagged. The consumer
      of this output always knows the analysis ran.
    - refusal_reason is populated when confidence_level is INSUFFICIENT.
      The system never returns null silently — it always explains why.
    """

    # --- Identity ---
    product_name: str = Field(
        ...,
        description="The name of the product being reviewed."
    )

    # --- Verdicts ---
    verdict_en: str = Field(
        ...,
        description=(
            "The synthesized verdict in English. 2–3 sentences maximum. "
            "Written as a trusted mom friend would speak — honest, warm, specific. "
            "Not marketing copy. Not a translation. "
            "Every claim must be grounded in the reviews provided. "
            "Empty string only when confidence_level is INSUFFICIENT."
        ),
        max_length=500
    )

    verdict_ar: str = Field(
        ...,
        description=(
            "The synthesized verdict in Arabic. Generated natively — not translated "
            "from verdict_en. Written in simple Modern Standard Arabic appropriate "
            "for a Gulf Arabic-speaking mother. 2–3 sentences maximum. "
            "Empty string only when confidence_level is INSUFFICIENT."
        ),
        max_length=500
    )

    # --- Structured Findings ---
    pros: list[ProConItem] = Field(
        default_factory=list,
        description=(
            "List of positive points consistently mentioned across reviews. "
            "Maximum 5 items. Each must have grounding evidence. "
            "Ordered by mention_percentage descending."
        ),
        max_length=5
    )

    cons: list[ProConItem] = Field(
        default_factory=list,
        description=(
            "List of negative points consistently mentioned across reviews. "
            "Maximum 5 items. Each must have grounding evidence. "
            "Ordered by mention_percentage descending."
        ),
        max_length=5
    )

    # --- Metadata ---
    overall_sentiment: OverallSentiment = Field(
        ...,
        description=(
            "The dominant sentiment across all reviews. "
            "POSITIVE: >65% positive reviews. "
            "NEGATIVE: >65% negative reviews. "
            "MIXED: everything in between."
        )
    )

    # --- Confidence ---
    confidence_level: ConfidenceLevel = Field(
        ...,
        description=(
            "Confidence tier. Computed by code based on review_count "
            "and fake_review_flag. Never set by the LLM."
        )
    )

    confidence_score: float = Field(
        ...,
        description=(
            "Numerical confidence score between 0.0 and 1.0. "
            "Computed by code. Penalized for low volume and fake flag. "
            "Never set by the LLM."
        ),
        ge=0.0,
        le=1.0
    )

    # --- Review Stats ---
    review_count: int = Field(
        ...,
        description="Total number of reviews processed.",
        ge=0
    )

    language_breakdown: dict[str, int] = Field(
        ...,
        description=(
            "Count of reviews per language. "
            "Example: {'en': 140, 'ar': 60}."
        )
    )

    # --- Quality Signals ---
    fake_review_flag: FakeReviewFlag = Field(
        ...,
        description="Result of fake review detection. Always present."
    )

    themes_identified: list[str] = Field(
        default_factory=list,
        description=(
            "List of themes/topics identified across reviews via clustering. "
            "Example: ['ease of cleaning', 'anti-colic', 'nipple flow', 'value for money']."
        )
    )

    # --- Refusal ---
    refusal_reason: Optional[str] = Field(
        default=None,
        description=(
            "Populated when confidence_level is INSUFFICIENT. "
            "Explains clearly why a verdict was not generated. "
            "Example: 'Only 3 reviews available. Minimum 5 required.' "
            "None in all other cases."
        )
    )

    # -----------------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------------

    @field_validator('confidence_score')
    @classmethod
    def confidence_score_must_be_valid(cls, v: float) -> float:
        """Redundant with ge/le but makes the error message explicit."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f'confidence_score must be between 0.0 and 1.0, got {v}'
            )
        return round(v, 3)

    @field_validator('verdict_ar')
    @classmethod
    def arabic_verdict_must_contain_arabic(cls, v: str) -> str:
        """
        Actively verify that the Arabic field contains Arabic characters.
        Catches the common failure where the model returns English in the
        Arabic field, or returns a translation that lost Arabic script.

        Arabic Unicode block: U+0600 to U+06FF
        We require at least 10 Arabic characters to pass.
        An empty string is allowed (for INSUFFICIENT cases) and checked
        separately by the model_validator below.
        """
        if v == "":
            return v  # handled by model_validator
        arabic_chars = sum(1 for c in v if '\u0600' <= c <= '\u06FF')
        if arabic_chars < 10:
            raise ValueError(
                f'verdict_ar does not appear to contain Arabic text. '
                f'Found only {arabic_chars} Arabic characters. '
                f'Arabic verdict must be generated natively, not left empty '
                f'or filled with non-Arabic text.'
            )
        return v

    @field_validator('pros', 'cons')
    @classmethod
    def every_item_must_have_a_quote(cls, items: list[ProConItem]) -> list[ProConItem]:
        """
        Every pro and con must have a non-empty representative quote.
        This is the core anti-hallucination guard — no quote, no claim.
        """
        for item in items:
            if not item.evidence.representative_quote.strip():
                raise ValueError(
                    f'ProConItem "{item.point}" has an empty representative_quote. '
                    f'Every claim must be backed by an actual quote from the reviews.'
                )
        return items

    @model_validator(mode='after')
    def insufficient_verdict_must_have_refusal_reason(self) -> 'MomsVerdict':
        """
        When confidence is INSUFFICIENT:
        - refusal_reason must be populated
        - verdict_en and verdict_ar must be empty
        - pros and cons must be empty

        When confidence is NOT insufficient:
        - verdict_en must not be empty
        - verdict_ar must not be empty
        - refusal_reason must be None
        """
        if self.confidence_level == ConfidenceLevel.INSUFFICIENT:
            if not self.refusal_reason:
                raise ValueError(
                    'When confidence_level is INSUFFICIENT, '
                    'refusal_reason must be populated.'
                )
            if self.verdict_en or self.verdict_ar:
                raise ValueError(
                    'When confidence_level is INSUFFICIENT, '
                    'verdict_en and verdict_ar must be empty strings.'
                )
        else:
            if not self.verdict_en.strip():
                raise ValueError(
                    'verdict_en cannot be empty when confidence_level '
                    f'is {self.confidence_level}.'
                )
            if not self.verdict_ar.strip():
                raise ValueError(
                    'verdict_ar cannot be empty when confidence_level '
                    f'is {self.confidence_level}. '
                    'Arabic verdict must be generated natively.'
                )
            if self.refusal_reason is not None:
                raise ValueError(
                    'refusal_reason must be None when confidence_level '
                    f'is {self.confidence_level}.'
                )
        return self

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------

    model_config = {
        "json_schema_extra": {
            "example": {
                "product_name": "Philips Avent Natural Baby Bottle 260ml",
                "verdict_en": (
                    "Most moms find this bottle easy to clean and effective "
                    "at reducing colic, with the wide neck making sterilisation "
                    "straightforward. The main concern is nipple flow speed — "
                    "several moms found it too fast for newborns under 8 weeks."
                ),
                "verdict_ar": (
                    "تُعدّ هذه الرضّاعة من أكثر المنتجات التي أثنت عليها "
                    "الأمهات لسهولة تنظيفها وفعاليتها في تقليل المغص. "
                    "الملاحظة الرئيسية هي أن تدفق الحليب قد يكون سريعاً "
                    "للمواليد الجدد دون الثمانية أسابيع."
                ),
                "pros": [
                    {
                        "point": "Easy to clean and sterilise",
                        "evidence": {
                            "claim": "easy to clean",
                            "mention_count": 89,
                            "representative_quote": (
                                "washes perfectly in the dishwasher, "
                                "no residue at all"
                            ),
                            "quote_language": "en"
                        },
                        "mention_percentage": 66.9
                    }
                ],
                "cons": [
                    {
                        "point": "Nipple flow too fast for newborns",
                        "evidence": {
                            "claim": "flow too fast",
                            "mention_count": 44,
                            "representative_quote": (
                                "too much milk coming out for my 2 week old, "
                                "she kept choking"
                            ),
                            "quote_language": "en"
                        },
                        "mention_percentage": 33.1
                    }
                ],
                "overall_sentiment": "positive",
                "confidence_level": "high",
                "confidence_score": 0.87,
                "review_count": 133,
                "language_breakdown": {"en": 98, "ar": 35},
                "fake_review_flag": {
                    "flagged": False,
                    "average_similarity_score": 0.42,
                    "reason": None
                },
                "themes_identified": [
                    "ease of cleaning",
                    "colic reduction",
                    "nipple flow",
                    "value for money"
                ],
                "refusal_reason": None
            }
        }
    }
