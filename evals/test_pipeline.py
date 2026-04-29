"""
evals/test_pipeline.py

Pytest unit tests for individual pipeline stages.
Run with: pytest evals/test_pipeline.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.schema import (
    Review, MomsVerdict, ConfidenceLevel,
    OverallSentiment, FakeReviewFlag, ProConItem, SupportingEvidence
)
from src.stages.preprocessor import Preprocessor, detect_language
from src.stages.fake_detector import FakeReviewDetector
from src.stages.validator import Validator


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_review_valid():
    r = Review(text="Great bottle!", rating=5.0, language="en")
    assert r.text == "Great bottle!"
    assert r.rating == 5.0


def test_review_rejects_short_text():
    with pytest.raises(Exception):
        Review(text="Hi", rating=5.0, language="en")


def test_review_rejects_bad_rating():
    with pytest.raises(Exception):
        Review(text="Good bottle overall", rating=6.0, language="en")


def test_moms_verdict_insufficient_requires_refusal_reason():
    with pytest.raises(Exception):
        MomsVerdict(
            product_name="Test",
            verdict_en="",
            verdict_ar="",
            pros=[],
            cons=[],
            overall_sentiment=OverallSentiment.MIXED,
            confidence_level=ConfidenceLevel.INSUFFICIENT,
            confidence_score=0.0,
            review_count=2,
            language_breakdown={"en": 2},
            fake_review_flag=FakeReviewFlag(
                flagged=False,
                average_similarity_score=0.3
            ),
            themes_identified=[],
            refusal_reason=None  # this should fail
        )


def test_moms_verdict_insufficient_valid():
    v = MomsVerdict(
        product_name="Test",
        verdict_en="",
        verdict_ar="",
        pros=[],
        cons=[],
        overall_sentiment=OverallSentiment.MIXED,
        confidence_level=ConfidenceLevel.INSUFFICIENT,
        confidence_score=0.0,
        review_count=2,
        language_breakdown={"en": 2},
        fake_review_flag=FakeReviewFlag(
            flagged=False,
            average_similarity_score=0.3
        ),
        themes_identified=[],
        refusal_reason="Only 2 reviews available."
    )
    assert v.confidence_level == ConfidenceLevel.INSUFFICIENT
    assert v.refusal_reason is not None


def test_pro_con_requires_quote():
    with pytest.raises(Exception):
        evidence = SupportingEvidence(
            claim="easy to clean",
            mention_count=5,
            representative_quote="",  # empty — should fail
            quote_language="en"
        )
        item = ProConItem(
            point="Easy to clean",
            evidence=evidence,
            mention_percentage=10.0
        )
        # trigger the validator by building a verdict with this item
        MomsVerdict(
            product_name="Test",
            verdict_en="Good product.",
            verdict_ar="منتج جيد جداً للأطفال وسهل الاستخدام.",
            pros=[item],
            cons=[],
            overall_sentiment=OverallSentiment.POSITIVE,
            confidence_level=ConfidenceLevel.HIGH,
            confidence_score=0.9,
            review_count=60,
            language_breakdown={"en": 60},
            fake_review_flag=FakeReviewFlag(
                flagged=False,
                average_similarity_score=0.3
            ),
            themes_identified=["cleaning"]
        )


# ---------------------------------------------------------------------------
# Preprocessor tests
# ---------------------------------------------------------------------------

def test_language_detection_english():
    assert detect_language("This is a great bottle for my baby") == "en"


def test_language_detection_arabic():
    assert detect_language("هذه الرضّاعة رائعة جداً لطفلي") == "ar"


def test_preprocessor_drops_short_reviews():
    preprocessor = Preprocessor()
    raw = [
        {"text": "Hi", "rating": 5.0, "language": "en"},
        {"text": "Great bottle, my baby loves it very much.", "rating": 5.0, "language": "en"}
    ]
    clean, stats = preprocessor.run(raw)
    assert len(clean) == 1
    assert stats["total_dropped"] == 1


def test_preprocessor_clamps_rating():
    preprocessor = Preprocessor()
    raw = [{"text": "Good enough bottle for everyday use.", "rating": 9.0, "language": "en"}]
    clean, stats = preprocessor.run(raw)
    assert clean[0].rating == 5.0


def test_preprocessor_language_breakdown():
    preprocessor = Preprocessor()
    raw = [
        {"text": "Great bottle overall really happy with purchase.", "rating": 5.0, "language": "en"},
        {"text": "رضّاعة ممتازة وسهلة التنظيف جداً.", "rating": 5.0, "language": "ar"}
    ]
    clean, stats = preprocessor.run(raw)
    assert stats["language_breakdown"].get("en") == 1
    assert stats["language_breakdown"].get("ar") == 1


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

def test_validator_insufficient_on_low_count():
    validator = Validator()
    fake_flag = FakeReviewFlag(flagged=False, average_similarity_score=0.3)
    result = validator.run(
        extractions=[],
        review_count=3,
        fake_flag=fake_flag,
        language_breakdown={"en": 3}
    )
    assert result["confidence_level"] == ConfidenceLevel.INSUFFICIENT


def test_validator_fake_penalty_reduces_score():
    validator = Validator()
    fake_flag = FakeReviewFlag(flagged=True, average_similarity_score=0.92)
    result = validator.run(
        extractions=[],
        review_count=60,
        fake_flag=fake_flag,
        language_breakdown={"en": 60}
    )
    # base would be 0.90, penalty is 0.30, so should be 0.60
    assert result["confidence_score"] <= 0.65


def test_validator_high_confidence_large_dataset():
    validator = Validator()
    fake_flag = FakeReviewFlag(flagged=False, average_similarity_score=0.4)
    result = validator.run(
        extractions=[],
        review_count=100,
        fake_flag=fake_flag,
        language_breakdown={"en": 100}
    )
    assert result["confidence_level"] == ConfidenceLevel.HIGH
