"""
src/stages/clusterer.py

Stage 3: Theme clustering using KMeans on multilingual sentence embeddings.

Why we cluster before sending to the LLM:
- Sending 120 reviews in one prompt produces vague, hard-to-ground output.
  The model averages everything together and loses specificity.
- Clustering first means the LLM processes one topic at a time:
  "here are 34 reviews about cleaning — what do they say?"
  This produces more specific pros/cons with better quote grounding.
- It also reduces token cost — we send smaller, focused prompts.
- Theme names (used in MomsVerdict.themes_identified) come from this stage.

How it works:
1. Reuse embeddings already computed by FakeDetector (passed in, not recomputed).
   If embeddings are not passed in, compute them fresh.
2. Determine optimal cluster count using a simple heuristic based on review count.
3. Run KMeans clustering.
4. For each cluster, collect the reviews that belong to it.
5. Generate a human-readable theme label for each cluster using the
   most representative review (closest to cluster centroid).

Design decisions:
- KMeans over hierarchical clustering: simpler, faster, deterministic with fixed seed.
- We do NOT ask the LLM to name the themes — we derive labels from
  the most centroid-proximate review text. Cheaper and faster.
- Cluster count is capped at 5 — more than 5 themes makes the
  extraction prompt unwieldy and the output harder to read.
- Minimum cluster size is enforced — tiny clusters (< 3 reviews) are
  merged into the nearest larger cluster.

Limitations (for TRADEOFFS.md):
- KMeans assumes spherical clusters — not always true for review themes.
  In production, HDBSCAN would be more flexible.
- Theme labels are heuristic — they're the first sentence of the most
  central review, not a proper topic label. Good enough for a prototype.
- Fixed random seed (42) makes results reproducible but means the
  clustering won't adapt to very different data distributions.
"""

import os
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from src.schema import Review


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# cluster count boundaries
MIN_CLUSTERS = 2
MAX_CLUSTERS = 5

# clusters with fewer than this many reviews get merged into nearest cluster
MIN_CLUSTER_SIZE = 3

# KMeans random seed — fixed for reproducibility
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReviewCluster:
    """
    A group of thematically related reviews.

    Attributes:
        cluster_id: Integer cluster index from KMeans
        theme_label: Human-readable label derived from most central review
        reviews: All Review objects assigned to this cluster
        centroid: The cluster centroid vector (used for finding representative review)
        representative_review: The review closest to the centroid
    """
    cluster_id: int
    theme_label: str
    reviews: list[Review]
    centroid: np.ndarray
    representative_review: Review

    def __repr__(self):
        return (
            f"ReviewCluster(id={self.cluster_id}, "
            f"theme='{self.theme_label}', "
            f"n_reviews={len(self.reviews)})"
        )


# ---------------------------------------------------------------------------
# Clusterer
# ---------------------------------------------------------------------------

class Clusterer:
    """
    Groups reviews into thematic clusters using KMeans.

    Usage:
        clusterer = Clusterer()
        clusters = clusterer.run(reviews)
        for cluster in clusters:
            print(cluster.theme_label, len(cluster.reviews))
    """

    def __init__(self):
        self._model = None  # lazy load, same pattern as FakeDetector

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"  [Clusterer] Loading embedding model: {EMBEDDING_MODEL_NAME}")
            self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print(f"  [Clusterer] Model loaded.")
        return self._model

    def run(
        self,
        reviews: list[Review],
        embeddings: np.ndarray | None = None
    ) -> list[ReviewCluster]:
        """
        Cluster reviews into thematic groups.

        Args:
            reviews: List of validated Review objects (from Preprocessor)
            embeddings: Optional pre-computed embeddings from FakeDetector.
                        If provided, we skip recomputing them (saves time).
                        Must be same order as reviews list.

        Returns:
            List of ReviewCluster objects, sorted by size descending.
        """

        n = len(reviews)

        # --- Gate: too few reviews to cluster meaningfully ---
        if n < MIN_CLUSTERS * MIN_CLUSTER_SIZE:
            # return everything as one cluster
            print(
                f"  [Clusterer] Only {n} reviews — returning single cluster "
                f"(min needed for multi-cluster: "
                f"{MIN_CLUSTERS * MIN_CLUSTER_SIZE})"
            )
            return self._single_cluster(reviews)

        # --- Compute or reuse embeddings ---
        if embeddings is None:
            print(f"  [Clusterer] Computing embeddings for {n} reviews...")
            embeddings = self.model.encode(
                [r.text for r in reviews],
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        else:
            print(f"  [Clusterer] Reusing {len(embeddings)} pre-computed embeddings.")

        # --- Determine cluster count ---
        n_clusters = self._determine_cluster_count(n)
        print(f"  [Clusterer] Using {n_clusters} clusters for {n} reviews.")

        # --- Run KMeans ---
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_SEED,
            n_init=10,      # run 10 times with different seeds, pick best
            max_iter=300
        )
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        # --- Build clusters ---
        raw_clusters = self._build_clusters(
            reviews=reviews,
            embeddings=embeddings,
            labels=labels,
            centroids=centroids
        )

        # --- Merge tiny clusters ---
        clusters = self._merge_tiny_clusters(raw_clusters, embeddings, labels)

        # --- Sort by size descending (largest theme first) ---
        clusters.sort(key=lambda c: len(c.reviews), reverse=True)

        # --- Report ---
        print(f"  [Clusterer] Final clusters:")
        for cluster in clusters:
            print(
                f"    [{cluster.cluster_id}] '{cluster.theme_label}' "
                f"— {len(cluster.reviews)} reviews"
            )

        return clusters

    def _determine_cluster_count(self, n_reviews: int) -> int:
        """
        Heuristic: more reviews → more clusters, but capped at MAX_CLUSTERS.

        Breakpoints chosen so each cluster has at least ~15 reviews on average:
        - < 20 reviews  → 2 clusters
        - 20–49 reviews → 3 clusters
        - 50–99 reviews → 4 clusters
        - 100+ reviews  → 5 clusters
        """
        if n_reviews < 20:
            return 2
        elif n_reviews < 50:
            return 3
        elif n_reviews < 100:
            return 4
        else:
            return MAX_CLUSTERS

    def _build_clusters(
        self,
        reviews: list[Review],
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ) -> list[ReviewCluster]:
        """
        Build ReviewCluster objects from KMeans output.

        For each cluster:
        1. Collect all reviews assigned to it.
        2. Find the review closest to the centroid (most representative).
        3. Derive a theme label from that review's first sentence.
        """
        n_clusters = len(centroids)
        clusters = []

        for cluster_id in range(n_clusters):
            # find which reviews belong to this cluster
            mask = labels == cluster_id
            cluster_indices = np.where(mask)[0]

            if len(cluster_indices) == 0:
                continue

            cluster_reviews = [reviews[i] for i in cluster_indices]
            cluster_embeddings = embeddings[cluster_indices]

            # find the most representative review —
            # the one with minimum distance to the centroid
            centroid = centroids[cluster_id]
            distances = pairwise_distances(
                cluster_embeddings,
                centroid.reshape(1, -1),
                metric='cosine'
            ).flatten()
            most_central_idx = int(np.argmin(distances))
            representative_review = cluster_reviews[most_central_idx]

            # derive theme label from representative review
            theme_label = self._derive_theme_label(representative_review)

            clusters.append(ReviewCluster(
                cluster_id=cluster_id,
                theme_label=theme_label,
                reviews=cluster_reviews,
                centroid=centroid,
                representative_review=representative_review
            ))

        return clusters

    def _derive_theme_label(self, review: Review) -> str:
        """
        Derive a short theme label from a review.

        Strategy: take the first sentence of the most central review,
        truncate to 60 characters. This is a heuristic — it won't always
        produce perfect labels, but it's fast, free, and good enough
        for logging and the themes_identified field.

        In production you'd use a short LLM call to name the theme properly.
        We document this tradeoff in TRADEOFFS.md.
        """
        text = review.text.strip()

        # get first sentence
        for delimiter in ['. ', '! ', '? ', '.\n', '،', '。']:
            if delimiter in text:
                first_sentence = text.split(delimiter)[0].strip()
                break
        else:
            first_sentence = text

        # truncate
        label = first_sentence[:60]
        if len(first_sentence) > 60:
            label += "..."

        return label

    def _merge_tiny_clusters(
        self,
        clusters: list[ReviewCluster],
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> list[ReviewCluster]:
        """
        Merge clusters smaller than MIN_CLUSTER_SIZE into the nearest
        larger cluster (by centroid distance).

        This prevents tiny clusters (1-2 reviews) from becoming their own
        theme, which would produce low-quality pros/cons with minimal evidence.
        """
        large = [c for c in clusters if len(c.reviews) >= MIN_CLUSTER_SIZE]
        small = [c for c in clusters if len(c.reviews) < MIN_CLUSTER_SIZE]

        if not small:
            return clusters  # nothing to merge

        if not large:
            # all clusters are tiny — just return them all, nothing to merge into
            return clusters

        print(
            f"  [Clusterer] Merging {len(small)} tiny cluster(s) "
            f"into nearest large cluster."
        )

        for tiny_cluster in small:
            # find the large cluster whose centroid is closest to this tiny one
            best_large = None
            best_distance = float('inf')

            for large_cluster in large:
                dist = float(pairwise_distances(
                    tiny_cluster.centroid.reshape(1, -1),
                    large_cluster.centroid.reshape(1, -1),
                    metric='cosine'
                )[0][0])

                if dist < best_distance:
                    best_distance = dist
                    best_large = large_cluster

            if best_large is not None:
                # merge tiny cluster's reviews into the nearest large cluster
                best_large.reviews.extend(tiny_cluster.reviews)
                print(
                    f"  [Clusterer] Merged cluster {tiny_cluster.cluster_id} "
                    f"({len(tiny_cluster.reviews)} reviews) → "
                    f"cluster {best_large.cluster_id} '{best_large.theme_label}'"
                )

        return large

    def _single_cluster(self, reviews: list[Review]) -> list[ReviewCluster]:
        """
        Fallback: return all reviews as a single cluster.
        Used when there are too few reviews to cluster meaningfully.
        """
        if not reviews:
            return []

        # use first review as representative (no embeddings computed)
        representative = reviews[0]
        theme_label = self._derive_theme_label(representative)

        # create a fake centroid (zeros) — won't be used downstream
        dummy_centroid = np.zeros(384)  # MiniLM embedding dim is 384

        return [ReviewCluster(
            cluster_id=0,
            theme_label=theme_label,
            reviews=reviews,
            centroid=dummy_centroid,
            representative_review=representative
        )]
