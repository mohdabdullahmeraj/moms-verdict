# Architectural Tradeoffs

Every engineering decision involves tradeoffs. This document outlines the intentional compromises made while designing the Moms Verdict AI pipeline, prioritizing reliability, determinism, and trust over raw LLM reliance.

## 1. KMeans Clustering vs. HDBSCAN
**Decision:** We use KMeans (`sklearn.cluster.KMeans`) to cluster review themes prior to LLM extraction, rather than a density-based algorithm like HDBSCAN.

**Tradeoff:** 
KMeans is fast, simple, and deterministic (with a fixed random seed), but it assumes clusters are roughly spherical and of similar variance. Review text embeddings in high-dimensional space often form irregular density manifolds rather than clean spheres. 

**Why we made it:**
For a prototype, KMeans is sufficient to chunk 100+ reviews into 3-5 distinct semantic buckets. This successfully prevents the LLM "lost in the middle" hallucination problem without requiring heavy tuning of `min_cluster_size` or `min_samples` parameters that density-based clustering requires. In a production environment with varying dataset sizes (from 50 to 5,000 reviews), HDBSCAN would be the superior, though more computationally expensive, choice.

## 2. Deterministic Confidence vs. LLM-Assessed Confidence
**Decision:** The `ConfidenceLevel` and `confidence_score` are calculated using deterministic Python logic in `Validator` based on sample volume, fake-review penalties, and rating-text mismatch rates. We do *not* ask the LLM to output a "confidence score".

**Tradeoff:**
The deterministic model is rigid. It cannot intuitively "feel" if a set of reviews is highly contradictory or nuanced in ways a neural network might detect.

**Why we made it:**
LLMs are notoriously bad at self-assessing confidence. They tend to be overly confident (outputting `0.95` arbitrarily) or overly cautious. By moving this logic into a mechanical layer, we guarantee that a product with 4 reviews will *never* receive a `HIGH` confidence score, regardless of what the LLM generates. This is critical for consumer trust.

## 3. Recomputing Embeddings
**Decision:** We compute local embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) inside the `FakeDetector` stage. We then cache them and pass them to the `Clusterer` stage. If the cache fails or isn't passed, the `Clusterer` recomputes them.

**Tradeoff:**
Passing floating-point matrices between modular stages breaks strict isolation slightly. If we didn't pass them, we would waste CPU cycles recomputing the exact same vectors.

**Why we made it:**
It strikes a balance between performance and modularity. The pipeline orchestrator handles the handoff. In a microservice production architecture, these embeddings would likely be computed once upon ingestion and stored in a vector database (e.g., Pinecone or pgvector), and both stages would simply query the DB.

## 4. Native Arabic Generation vs. Translation
**Decision:** The English verdict is constructed deterministically via a string template. The Arabic verdict is generated natively from scratch by a dedicated Gemini Pro call, using the structured data (Pros/Cons) as input. We do not translate the English output into Arabic.

**Tradeoff:**
We spend an extra LLM API call (and associated latency/cost) specifically for the Arabic generation. Furthermore, the English and Arabic verdicts might diverge slightly in tone or specific phrasing.

**Why we made it:**
For a GCC-focused e-commerce platform, translated copy is a massive trust-breaker. Translated text often uses English syntax mapped to Arabic words, ignoring the register and cultural idioms of the target demographic (Gulf mothers). By passing raw, language-agnostic facts to the LLM and prompting it in Arabic to write as a native speaker, we achieve vastly superior linguistic quality. The minor cost of an extra API call is negligible compared to the increase in user trust.

## 5. Single LLM Call per Cluster
**Decision:** Instead of sending all 120 reviews to the LLM at once, we break them into clusters and make one LLM call per cluster to extract pros and cons.

**Tradeoff:**
More API calls equals higher latency and higher cost. If one cluster's API call fails due to network issues, that theme's data is missing from the final assembly.

**Why we made it:**
Stuffing an entire dataset into a single prompt causes the model to "average out" the nuance and hallucinate non-specific quotes. By chunking the data semantically, we constrain the LLM to a single topic (e.g., "focus *only* on these 30 reviews about cleaning"). This drastically improves the specificity of the claims and guarantees high-fidelity quote grounding. We handle failures gracefully by allowing the pipeline to assemble the final verdict from whatever clusters successfully resolved.
