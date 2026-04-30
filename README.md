# 🌸 Moms Verdict

An end-to-end, deterministic AI pipeline that synthesizes raw e-commerce reviews into structured, grounded, and bilingual (English/Arabic) product verdicts.

---

## What Is the Problem 

Imagine you're a mom. You go to a shopping website to buy a baby bottle. The bottle has 340 reviews. You don't have time to read 340 reviews — you have a baby crying in the next room. So you read maybe 3-4, feel uncertain, and either buy hesitantly or just leave.

Now imagine instead, right below the product, there's a box that says:

> "Moms love how easy this is to clean and how well the anti-colic design works — 78% of reviewers mentioned this. The main complaint is that the nipple flow is too fast for newborns, mentioned by 34 moms. Overall: great for babies 3 months and up, with some reservations for newborns. Based on 340 reviews."

And below that, the same thing in Arabic, written naturally — not like a Google Translate output, but like a real Arabic-speaking mom wrote it.

That box — that's what we're building. We're building the system that reads all the reviews, figures out what moms actually care about, and writes that summary automatically. In English and Arabic.

## Why Is This Hard

Three reasons:

1. **First, AI makes stuff up.** If you just ask an AI "summarize these reviews," it will confidently say things that no review actually said. It might say "moms love the BPA-free material" when zero reviews mentioned BPA. That's dangerous on a shopping site — it's a false trust signal. Our system is built so every single claim must be traceable to an actual review quote. If it can't find a quote, it doesn't make the claim.
2. **Second, Arabic is not just translated English.** If you write the verdict in English and then run it through a translator, any Arabic speaker immediately knows. The sentence structure is wrong, the tone is wrong, the idioms are wrong. We generate the Arabic verdict separately, natively — we give the AI the facts and ask it to write in Arabic the way an Arabic-speaking Gulf mom would actually write.
3. **Third, not all reviews are trustworthy.** Some products have fake reviews — 50 reviews that all sound suspiciously similar, clearly written by the same person or a bot. If you summarize fake reviews, you produce a fake verdict. Our system detects when reviews are too similar to each other and automatically flags this and reduces its own confidence.

---

## Tech Stack

- **Orchestration:** Python 3.10+
- **LLM Provider:** OpenRouter (using `openai/gpt-oss-20b:free`) via the `openai` SDK.
- **Embeddings:** Local `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Clustering:** `scikit-learn` (KMeans)
- **Data Validation:** `pydantic` v2
- **Interface:** `gradio`

## Architecture Pipeline

The system is fully decoupled into 7 distinct sequential stages. It uses deterministic code for routing and validation, only relying on LLMs for extraction and native generation.

```text
[ Raw Reviews JSON ]
   ↓
[ Stage 1 — Preprocessor ]
(Clean · Validate · Detect Language)
   ↓
[ Stage 2 — Fake Detector ]
(Embeddings · Cosine Similarity · Spam Flag)
   ↓
[ Stage 3 — Clusterer ]
(KMeans · Thematic Groups)
   ↓
[ Stage 4 — Extractor ]
(OpenRouter · JSON Mode · One call per cluster)
   ↓
[ Stage 5 — Validator ]
(Confidence computed in code · Deduplicate)
   ↓
[ Stage 6 — Arabic Generator ]
(Arabic prompt in Arabic · Retry 3x)
   ↓
[ Stage 7 — Assembly ]
(Schema validates · Invalid states rejected)
   ↓
[ MomsVerdict — Pydantic Validated ]
```

---

## Setup & Run Instructions
*Under 5 minutes from clone to first output.*

First, clone the repository:
```bash
git clone https://github.com/mohdabdullahmeraj/moms-verdict.git
cd moms-verdict
```

### Option 1: Local / Virtual Environment
1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows: venv\\Scripts\\activate
   # On macOS/Linux: source venv/bin/activate
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set your API Key:**
   Create a `.env` file in the root directory (or copy `.env.example`):
   ```env
   OPENROUTER_API_KEY="your_openrouter_api_key_here"
   ```
4. **Run the app:**
   ```bash
   python app.py
   ```
   *Visit `http://127.0.0.1:7860` in your browser.*

### Option 2: Docker
If you prefer containerization, you can run the entire stack (including the UI and background evaluation jobs) via Docker Compose.

1. **Set your API Key:**
   Ensure you have a `.env` file with your `OPENROUTER_API_KEY`.
2. **Spin up the stack:**
   ```bash
   docker-compose up --build
   ```
3. **Access the App:**
   *Visit `http://localhost:7860` in your browser.*

### Option 3: Google Colab (Recommended for zero-setup)
1. Open a new Google Colab notebook.
2. In the first cell, clone the repo and install dependencies:
   ```python
   !git clone https://github.com/mohdabdullahmeraj/moms-verdict.git
   %cd moms-verdict
   !pip install -r requirements.txt
   ```
3. Add your `OPENROUTER_API_KEY` in the **Colab Secrets** tab (the key icon on the left sidebar).
4. Run the app:
   ```python
   !python app.py
   ```
   *Click the public `*.gradio.live` link generated in the output.*

---

## Model & Architecture Choices

| Decision | Choice | Rationale |
|---|---|---|
| Extraction LLM | `openai/gpt-oss-20b:free` via OpenRouter | Free, reliable, strong JSON mode |
| Arabic generation | Same model, separate prompt | Cost-efficient at prototype scale |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Free, local, supports EN+AR natively |
| Clustering | KMeans (scikit-learn) | Deterministic, no API, reproducible |
| Confidence scoring | Pure Python code | LLMs are unreliable at self-assessment |
| Arabic prompt language | Written in Arabic | Produces measurably more natural output |
| English verdict | Template, no LLM | Deterministic, testable, saves API calls |
| Schema | Pydantic v2 with cross-field validators | Invalid states unrepresentable |

---

## Handling Uncertainty

The system has three explicit uncertainty mechanisms:

**1. Refusal** — fewer than 5 reviews → `INSUFFICIENT` confidence,
empty verdicts, `refusal_reason` populated. No verdict generated.

**2. Confidence tiers** — computed mechanically:
- HIGH: 50+ reviews, not flagged → base 0.90
- MEDIUM: 15–49 reviews → base 0.72
- LOW: 5–14 reviews → base 0.50
- Fake flag penalty: −0.30
- Rating-text mismatch penalty: −0.002 per 1%

**3. Fake review flag** — average pairwise embedding similarity
above threshold triggers flag, reduces confidence, and surfaces
a warning in the UI. The LLM is never told about this — the
safeguard is entirely mechanical.

---

## Evals

### Rubric (per test case, max 8 points)

| Criterion | Points | What We Check |
|---|---|---|
| Schema validity | 0–2 | Pydantic validates without error |
| Confidence correct | 0–2 | Level matches expected for review count |
| Fake flag correct | 0–1 | Detector flags/passes correctly |
| Refusal correct | 0–1 | Refuses when < 5 reviews |
| No hallucination | 0–1 | Verdict doesn't mention absent claims |
| Arabic valid | 0–1 | verdict_ar contains Arabic Unicode |

### Test Cases (12 total)

| ID | Name | Description | Score |
|---|---|---|---|
| TC01 | Happy path — large English dataset | Large genuine EN+AR dataset. Should produce high confidence positive verdict. | 8/8 |
| TC02 | Mixed sentiment — stroller | Mixed dataset. Should surface both positives and weight/bulk complaints. | 7/8 |
| TC03 | Fake review detection | Suspiciously similar reviews. Fake detector must flag this dataset. | 7/8 |
| TC04 | Insufficient reviews — refusal | Only 3 reviews. Pipeline must refuse and populate refusal_reason. | 8/8 |
| TC05 | Hallucination check | Reviews never explicitly mention BPA-free. Verdict must not hallucinate this claim. | 8/8 |
| TC06 | Empty reviews list | Empty input. Pipeline must return INSUFFICIENT, not crash. | 8/8 |
| TC07 | Borderline volume | 6 reviews — just above minimum threshold. Should produce LOW confidence verdict. | 6/8 |
| TC08 | Rating-text mismatch detection | All 5-star ratings but text is negative. Preprocessor should catch rating-text mismatches. | 7/8 |
| TC09 | Native Arabic reviews only | All Arabic reviews. Arabic verdict must be generated natively and contain Arabic text. | 6/8 |
| TC10 | Contradictory reviews | Reviews directly contradict on nipple flow. Should appear in both pros and cons. | 7/8 |
| TC11 | Very short reviews edge case | All reviews are very short (under 5 words). Pipeline should still work without crashing. | 6/8 |
| TC12 | High volume — theme richness | Large dataset should produce 3+ themes identified covering different topics. | 8/8 |

### Final Scorecard

```text
=================================================================
SCORECARD
=================================================================
  [TC01] ████████ 8/8 — Happy path — large English dataset
  [TC02] ███████░ 7/8 — Mixed sentiment — stroller
  [TC03] ███████░ 7/8 — Fake review detection
  [TC04] ████████ 8/8 — Insufficient reviews — refusal
  [TC05] ████████ 8/8 — Hallucination check — BPA-free not mentioned
  [TC06] ████████ 8/8 — Empty reviews list — explicit error
  [TC07] ██████░░ 6/8 — Borderline volume — 6 reviews
  [TC08] ███████░ 7/8 — Rating-text mismatch detection
  [TC09] ██████░░ 6/8 — Native Arabic reviews only
  [TC10] ███████░ 7/8 — Contradictory reviews on same topic
  [TC11] ██████░░ 6/8 — Very short reviews edge case
  [TC12] ████████ 8/8 — High volume — theme richness

Overall: 86/96 (89.6%)
Rating: STRONG
=================================================================
```

*Run `python evals/eval_runner.py` for full live scores and logs, or check `EVALS.md` for the complete execution logs of this run.*

### Honest Assessment

**What works well:**
- Fake detection correctly identifies spam datasets (similarity 0.93 caught)
- Refusal logic is clean — never generates a verdict with < 5 reviews
- Arabic generation produces natural Gulf-register text consistently
- Schema validation catches every structural error before output

**Known failure modes:**
- TC02 stroller scored MEDIUM not HIGH confidence (30 reviews,
  just below 50 threshold) — the threshold is conservative by design
- Fake detection catches similarity-based spam only, not
  diverse-but-coordinated fake reviews
- English verdict is template-based — loses nuance for edge case
  sentiment distributions
- Free LLM tier occasionally rate-limits, requiring model switching

---

## Tradeoffs

**Why confidence is computed in code, not by the LLM:**
LLMs produce confidence numbers that reflect training priors,
not actual data quality. Our score is a pure function of review
count, fake flag, and mismatch rate — deterministic and auditable.

**Why the Arabic prompt is written in Arabic:**
An English instruction saying "write in Arabic" produces
translated-sounding output. An Arabic instruction produces
native-register Gulf Arabic. Empirically verified on multiple runs.

**Why the English verdict uses a template:**
Arabic needs a model — grammar and register are too complex
for a template. English structured output from a template
is reliable, testable, and saves one API call per product.

**What was cut:**
- Docker (local venv and Colab sufficient at prototype scale)
- Real-time review ingestion (out of scope)
- Fine-tuning (no labelled data)
- Vector database (overkill for per-product batch processing)
- Full Mumzworld catalog integration

**What I would build next:**
- Per-category fake detection threshold tuning
- Confidence trend over time as reviews accumulate
- Webhook trigger to re-run pipeline when review count crosses threshold
- Fine-tuned Arabic model on Gulf-dialect product reviews

---

## Tooling

### Models and harnesses used

| Tool | Used for |
|---|---|
| Claude & Gemini (via Antigravity) | Architecture design, schema scaffolding, prompt iteration, debugging |
| `openai/gpt-oss-20b:free` via OpenRouter | Runtime extraction and Arabic generation |
| `paraphrase-multilingual-MiniLM-L12-v2` | Local embeddings for fake detection and clustering |
| Google Colab | Runtime environment |

### How AI was used

**Antigravity + Claude** was the primary development harness throughout.
The initial project scaffold, Pydantic schema, pipeline orchestrator,
and all stage files were generated via pair-coding sessions —
providing the full architecture context upfront and then building
file by file with Claude generating complete implementations.

**Specific AI contributions:**
- One-shot generation of the Pydantic schema with cross-field
  validators (the `model_validator` tying INSUFFICIENT logic together)
- The Arabic prompt written in Arabic — Claude suggested this
  approach after explaining the register problem
- The fake detection algorithm (embedding cosine similarity) —
  Claude proposed this as a deterministic alternative to asking
  an LLM "are these reviews fake?"
- Eval test case design — Claude generated adversarial cases
  including the hallucination check and contradictory review tests

**Where I overruled the agent:**
- Claude initially suggested using the LLM to compute confidence scores.
  Overruled — confidence is computed entirely in code. LLMs are
  unreliable at self-assessment.
- Claude suggested translating the English verdict to Arabic.
  Overruled — Arabic is generated independently from structured facts
  using a prompt written in Arabic.
- Claude suggested Docker as the deployment target. Overruled in
  favour of Colab + venv for prototype scale, with Docker kept
  in the repo for documentation purposes.

**What worked:**
- Full agent loops for boilerplate (schema, preprocessor, clusterer)
  produced correct, well-structured code on first pass
- Prompt iteration for the Arabic generation was fast — 3 rounds
  of refinement to get natural Gulf register output

**What did not work:**
- Claude's initial extraction prompt produced hallucinated quotes.
  Required manual iteration to enforce the "no quote = no claim" rule
- Auto-generated eval runner missed edge cases (empty reviews,
  inline test data). Manually extended.

### Key Prompts that Materially Shaped Output

The most critical prompt in the system is the **Arabic Generation Prompt**. Rather than translating an English summary, we feed the LLM structured facts and instruct it entirely in Arabic. 

Empirically, providing instructions *in Arabic* activates the model's native language capabilities much better than instructing it in English to "write in Arabic".

Here is the exact prompt template used in `src/prompts/arabic_prompt.py`:

```arabic
أنتِ محررة محتوى متخصصة في منتجات الأمومة والطفولة لموقع عالم ماما،
أكبر منصة تسوق إلكترونية للأمهات في منطقة الشرق الأوسط.

مهمتك: اكتبي ملخصاً موجزاً بالعربية عن آراء الأمهات في هذا المنتج،
بناءً على البيانات المستخرجة من التقييمات الفعلية أدناه.

---
المنتج: {product_name}
عدد التقييمات التي تم تحليلها: {review_count}
الانطباع العام: {sentiment_ar}
مستوى الثقة بالنتائج: {confidence_ar}
توزيع اللغات: {language_breakdown_ar}

الإيجابيات المذكورة في التقييمات:
{pros_ar}

السلبيات المذكورة في التقييمات:
{cons_ar}
---

تعليمات الكتابة — اقرئيها بعناية:

١. اكتبي بالعربية الفصحى المبسطة المناسبة للقارئة الخليجية.
   أسلوب صديقة أم تنصح صديقتها — صادق، دافئ، محدد.

٢. جملتان أو ثلاث جمل كحد أقصى.
   المحتوى يظهر على صفحة المنتج ويجب أن يكون سريع القراءة.

٣. لا تترجمي من الإنجليزية. اكتبي بشكل طبيعي كما تكتب أم عربية.
   التراكيب الإنجليزية المترجمة واضحة وتقلل من المصداقية.

٤. لا تضيفي معلومات غير موجودة في البيانات أعلاه.
   كل جملة تكتبينها يجب أن تعكس ما ذكره الأمهات فعلاً في تقييماتهن.

٥. إذا كان هناك سلبيات مهمة، اذكريها بصدق ولكن بأسلوب متوازن.
   الأمهات يثقن بالمحتوى الصادق أكثر من المدح المبالغ فيه.

٦. لا تبدئي الجملة بـ "هذا المنتج" — ابدئي بما يهم الأم مباشرة.

أعيدي النص العربي فقط، بدون أي تنسيق إضافي أو علامات أو شرح.
```

**Why this prompt works:**
1. **Gulf Register Specification (Rule 1):** Explicitly requests simplified Modern Standard Arabic tailored for a Gulf reader ("للقارئة الخليجية"). This formal-yet-warm register is critical for GCC e-commerce.
2. **Anti-Translation Instruction (Rule 3):** Explicitly forbids translating from English ("لا تترجمي من الإنجليزية"). Without this, LLMs often default to translated sentence structures even when given structured data.
3. **Data-Grounded (Rule 4):** Strictly bounds the model to only write sentences that reflect the exact injected `pros_ar` and `cons_ar` variables.

---

## Running Evals

```bash
# full eval runner (12 test cases)
python evals/eval_runner.py

# unit tests
pytest evals/test_pipeline.py -v
```

---

## Time Log

| Phase | Time |
|---|---|
| Problem selection + architecture | 45 min |
| Schema + pipeline design | 60 min |
| Core stages (preprocessor, detector, clusterer) | 90 min |
| LLM stages (extractor, Arabic generator) | 75 min |
| Gradio UI | 45 min |
| Evals (12 test cases) | 60 min |
| Documentation | 45 min |
| **Total** | **~7 hrs** |
