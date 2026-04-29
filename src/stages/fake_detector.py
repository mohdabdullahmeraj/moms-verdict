"""
src/stages/fake_detector.py

Fake review detection using multilingual sentence embeddings and
pairwise cosine similarity.

Design philosophy:
- No LLM involved. This is deterministic, reproducible, and fast.
- Uses sentence-transformers locally — no API calls, no cost, no rate limits.
- The embedding model (paraphrase-multilingual-MiniLM-L12-v2) supports
  Arabic and English natively, so mixed-language review sets work correctly.

How it works:
1. Embed every review text into a vector using the multilingual model.
2. Compute pairwise cosine similarity between all review pairs.
3. Average the similarity scores (excluding self-similarity on the diagonal).
4. If the average exceeds SUSPICION_THRESHOLD, flag the dataset.
5. Return a FakeReviewFlag with the score, flag, and reason.

Why cosine similarity and not something else:
- Cosine similarity measures direction, not magnitude — a long review and
  a short review saying the same thing will score high, which is exactly
  what we want to catch.
- Pairwise average across all pairs catches distributed similarity —
  it's not fooled by a few outlier reviews being similar while the rest
  are not.

Thresholds (set in .env):
- SUSPICION_THRESHOLD=0.85 — tuned empirically.
  Genuine review sets typically score 0.40–0.70.
  Coordinated fake sets typically score 0.85+.
  Adjust in .env if you find false positives on legitimate products.

Limitations (documented honestly for TRADEOFFS.md):
- Does not detect fake reviews that are genuinely diverse in wording
  but all follow the same rating pattern (all 5 stars, no complaints).
  That would require a separate rating-distribution check.
- Similarity threshold may need tuning per product category —
  very niche products may have naturally similar reviews
  (everyone mentions the same specific feature).
- Runs in O(n²) time — fine for hundreds of reviews, would need
  batching for thousands.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from src.schema import Review, FakeReviewFlag

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# the multilingual model — supports EN, AR, and 50+ other languages
# downloaded at Docker build time so startup is instant
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# above this average pairwise cosine similarity, we flag the review set
DEFAULT_SUSPICION_THRESHOLD = float(
    os.environ.get("SUSPICION_THRESHOLD", "0.85")
)

# below this many reviews, skip fake detection entirely
# (you can't meaningfully compute pairwise similarity on 2 reviews)
MIN_REVIEWS_FOR_DETECTION = 5


# ---------------------------------------------------------------------------
# FakeReviewDetector
# ---------------------------------------------------------------------------

class FakeReviewDetector:
    """
    Detects coordinated fake or spam reviews using embedding similarity.

    Usage:
        detector = FakeReviewDetector()
        flag = detector.run(reviews)
        if flag.flagged:
            print(f"Suspicious! Score: {flag.average_similarity_score}")
    """

    def __init__(self, suspicion_threshold: float = DEFAULT_SUSPICION_THRESHOLD):
        self.suspicion_threshold = suspicion_threshold
        self._model = None  # lazy load — don't load until first use

    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy load the embedding model.
        Loading takes ~2 seconds on first call.
        Subsequent calls reuse the loaded model.
        """
        if self._model is None:
            print(f"  [FakeDetector] Loading embedding model: {EMBEDDING_MODEL_NAME}")
            self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print(f"  [FakeDetector] Model loaded.")
        return self._model

    def run(self, reviews: list[Review]) -> FakeReviewFlag:
        """
        Main entry point. Takes validated Review objects, returns FakeReviewFlag.

        Args:
            reviews: List of validated Review objects from schema.py

        Returns:
            FakeReviewFlag with flagged status, similarity score, and reason
        """

        # --- Gate: too few reviews to analyze ---
        if len(reviews) < MIN_REVIEWS_FOR_DETECTION:
            # not enough reviews to compute meaningful similarity
            # return a neutral flag — this is not a fake detection result,
            # it's an insufficient data result
            return FakeReviewFlag(
                flagged=False,
                average_similarity_score=0.0,
                reason=(
                    f"Fake detection skipped: only {len(reviews)} reviews available. "
                    f"Minimum {MIN_REVIEWS_FOR_DETECTION} required for analysis."
                )
            )

        # --- Extract text from reviews ---
        texts = [review.text for review in reviews]

        # --- Compute embeddings ---
        # encode() returns a numpy array of shape (n_reviews, embedding_dim)
        # the multilingual model handles mixed EN/AR input correctly
        print(f"  [FakeDetector] Embedding {len(texts)} reviews...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,         # process in batches for memory efficiency
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # --- Compute pairwise cosine similarity ---
        avg_similarity = self._compute_average_pairwise_similarity(embeddings)
        print(f"  [FakeDetector] Average pairwise similarity: {avg_similarity:.4f}")

        # --- Determine flag ---
        flagged = avg_similarity > self.suspicion_threshold

        # --- Build reason string ---
        if flagged:
            reason = (
                f"Reviews show unusually high similarity "
                f"(average pairwise cosine similarity: {avg_similarity:.3f}, "
                f"threshold: {self.suspicion_threshold}). "
                f"This may indicate coordinated fake or spam reviews. "
                f"Verdict confidence has been automatically reduced."
            )
        else:
            reason = None

        return FakeReviewFlag(
            flagged=flagged,
            average_similarity_score=round(float(avg_similarity), 4),
            reason=reason
        )

    def _compute_average_pairwise_similarity(
        self,
        embeddings: np.ndarray
    ) -> float:
        """
        Compute the average pairwise cosine similarity across all review pairs.

        Steps:
        1. L2-normalize each embedding vector (so dot product = cosine similarity)
        2. Compute the full similarity matrix via matrix multiplication
        3. Exclude the diagonal (self-similarity is always 1.0, would inflate score)
        4. Return the mean of all off-diagonal values

        Args:
            embeddings: numpy array of shape (n_reviews, embedding_dim)

        Returns:
            float: average pairwise cosine similarity, range [0.0, 1.0]
        """
        n = len(embeddings)

        if n < 2:
            # can't compute pairwise similarity with one review
            return 0.0

        # L2 normalize: divide each vector by its magnitude
        # after this, dot product between any two vectors = cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        # avoid division by zero for zero vectors (shouldn't happen with real text)
        norms = np.where(norms == 0, 1e-8, norms)
        normalized = embeddings / norms

        # matrix multiplication: (n, dim) x (dim, n) = (n, n) similarity matrix
        # similarity_matrix[i][j] = cosine similarity between review i and review j
        similarity_matrix = np.dot(normalized, normalized.T)

        # create a mask that is False on the diagonal, True everywhere else
        # we exclude diagonal because similarity(review_i, review_i) = 1.0 always
        off_diagonal_mask = ~np.eye(n, dtype=bool)

        # extract off-diagonal values and compute mean
        off_diagonal_values = similarity_matrix[off_diagonal_mask]
        avg_similarity = float(off_diagonal_values.mean())

        return avg_similarity

    def get_most_similar_pairs(
        self,
        reviews: list[Review],
        top_n: int = 5
    ) -> list[dict]:
        """
        Returns the top N most similar review pairs.
        Useful for debugging — helps you see WHY a dataset was flagged.

        Args:
            reviews: List of validated Review objects
            top_n: How many pairs to return

        Returns:
            List of dicts with pair indices, similarity score, and review texts
        """
        if len(reviews) < 2:
            return []

        texts = [review.text for review in reviews]
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        # normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        normalized = embeddings / norms

        similarity_matrix = np.dot(normalized, normalized.T)

        # collect all unique pairs (upper triangle, excluding diagonal)
        pairs = []
        for i in range(len(reviews)):
            for j in range(i + 1, len(reviews)):
                pairs.append({
                    "index_a": i,
                    "index_b": j,
                    "similarity": float(similarity_matrix[i][j]),
                    "text_a": reviews[i].text[:100] + "..." if len(reviews[i].text) > 100 else reviews[i].text,
                    "text_b": reviews[j].text[:100] + "..." if len(reviews[j].text) > 100 else reviews[j].text,
                })

        # sort by similarity descending and return top N
        pairs.sort(key=lambda x: x["similarity"], reverse=True)
        return pairs[:top_n]
