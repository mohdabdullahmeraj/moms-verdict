"""
app.py - Moms Verdict UI
Clean dark theme inspired by modern dev tooling aesthetics.
"""

import json
import os
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import MomsVerdictPipeline
from src.schema import ConfidenceLevel

pipeline = MomsVerdictPipeline()
SAMPLE_DIR = Path("data/sample_products")


def load_sample_products():
    index_path = SAMPLE_DIR / "index.json"
    if not index_path.exists():
        return {}
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    return {item["product_name"]: item["file"] for item in index}


def format_pros_cons(pros, cons):
    lines = []
    if pros:
        lines.append("PROS")
        for p in pros:
            lines.append(
                f"  • {p.point}\n"
                f"    {p.evidence.mention_count} moms · {p.mention_percentage:.0f}% of reviews\n"
                f"    \"{p.evidence.representative_quote}\""
            )
    if cons:
        if lines:
            lines.append("")
        lines.append("CONS")
        for c in cons:
            lines.append(
                f"  • {c.point}\n"
                f"    {c.evidence.mention_count} moms · {c.mention_percentage:.0f}% of reviews\n"
                f"    \"{c.evidence.representative_quote}\""
            )
    return "\n\n".join(lines) if lines else "No pros/cons extracted."


def run_pipeline(sample_choice, custom_reviews_json, custom_product_name):
    try:
        if sample_choice and sample_choice != "── or paste your own reviews below ──":
            products = load_sample_products()
            filename = products.get(sample_choice)
            if not filename:
                return ("Sample not found.",) + ("",) * 6
            with open(SAMPLE_DIR / filename, encoding="utf-8") as f:
                data = json.load(f)
            product_name = data["product_name"]
            raw_reviews = data["reviews"]

        elif custom_reviews_json.strip():
            if not custom_product_name.strip():
                return ("Please enter a product name for custom reviews.",) + ("",) * 6
            try:
                raw_reviews = json.loads(custom_reviews_json)
            except json.JSONDecodeError as e:
                return (f"Invalid JSON: {e}",) + ("",) * 6
            product_name = custom_product_name.strip()

        else:
            return ("Please select a sample product or paste reviews JSON below.",) + ("",) * 6

        verdict = pipeline.run(product_name, raw_reviews)

        if verdict.confidence_level == ConfidenceLevel.INSUFFICIENT:
            return (
                f"⛔  Verdict refused\n\n{verdict.refusal_reason}",
                "—",
                "Insufficient data to generate pros & cons.",
                f"⛔  INSUFFICIENT  ·  {verdict.review_count} reviews processed",
                f"Similarity score: {verdict.fake_review_flag.average_similarity_score:.3f}",
                "None identified",
                verdict.model_dump_json(indent=2)
            )

        conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
            verdict.confidence_level.value, "⚪"
        )
        confidence_str = (
            f"{conf_emoji}  {verdict.confidence_level.value.upper()}  ·  "
            f"Score: {verdict.confidence_score}  ·  "
            f"{verdict.review_count} reviews  ·  "
            f"EN: {verdict.language_breakdown.get('en', 0)}  "
            f"AR: {verdict.language_breakdown.get('ar', 0)}"
        )

        fake_str = (
            f"⚠️  Flagged — {verdict.fake_review_flag.reason}"
            if verdict.fake_review_flag.flagged
            else f"✅  Genuine  ·  similarity score: {verdict.fake_review_flag.average_similarity_score:.3f}"
        )

        themes_str = "  ·  ".join(verdict.themes_identified) if verdict.themes_identified else "None"

        return (
            verdict.verdict_en,
            verdict.verdict_ar,
            format_pros_cons(verdict.pros, verdict.cons),
            confidence_str,
            fake_str,
            themes_str,
            verdict.model_dump_json(indent=2)
        )

    except Exception as e:
        import traceback
        return (f"Error: {e}", "", "", "", "", "", traceback.format_exc())


# ---------------------------------------------------------------------------
# Custom CSS — dark theme, Composio-inspired
# ---------------------------------------------------------------------------

CSS = """
/* ── Base ── */
body, .gradio-container {
    background-color: #f4f4f0 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #111111 !important;
}

/* ── Header ── */
.app-header {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 3px solid #111111;
    margin-bottom: 2rem;
}
.app-title {
    font-size: 2.5rem;
    font-weight: 900;
    color: #111111;
    text-transform: uppercase;
    letter-spacing: -1px;
    margin: 0;
}
.app-subtitle {
    font-size: 1rem;
    font-weight: 600;
    color: #111111;
    margin-top: 0.5rem;
}
.app-tagline {
    font-size: 0.9rem;
    color: #111111;
    font-weight: 700;
    margin-top: 0.5rem;
    background: #ffdb58; /* Mustard yellow accent */
    display: inline-block;
    padding: 4px 10px;
    border: 2px solid #111111;
    box-shadow: 2px 2px 0px #111111;
}

/* ── Cards ── */
.card {
    background: #ffffff !important;
    border: 3px solid #111111 !important;
    border-radius: 0px !important;
    box-shadow: 4px 4px 0px #111111 !important;
    padding: 1.25rem !important;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.8rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #111111;
    margin-bottom: 0.5rem;
    display: inline-block;
    border-bottom: 2px solid #111111;
}

/* ── Inputs ── */
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: #ffffff !important;
    border: 2px solid #111111 !important;
    border-radius: 0px !important;
    color: #111111 !important;
    font-size: 0.95rem !important;
    box-shadow: 2px 2px 0px #111111 !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
    outline: none !important;
    background: #f0f8ff !important;
}

/* ── Dropdown ── */
.gradio-container .wrap {
    background: #ffffff !important;
    border: 2px solid #111111 !important;
    border-radius: 0px !important;
    box-shadow: 2px 2px 0px #111111 !important;
}

/* ── Button ── */
.generate-btn {
    background: #ff5e5e !important;
    border: 3px solid #111111 !important;
    border-radius: 0px !important;
    color: #111111 !important;
    font-weight: 900 !important;
    font-size: 1rem !important;
    text-transform: uppercase !important;
    padding: 0.85rem 1.5rem !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 5px 5px 0px #111111 !important;
    transition: all 0.1s ease !important;
}
.generate-btn:active {
    box-shadow: 0px 0px 0px #111111 !important;
    transform: translate(5px, 5px) !important;
}

/* ── Output textboxes ── */
.gradio-container .output-textbox textarea {
    background: #ffffff !important;
    border: 2px solid #111111 !important;
    border-radius: 0px !important;
    box-shadow: 3px 3px 0px #111111 !important;
    color: #111111 !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    font-weight: 500 !important;
}

/* ── Arabic field ── */
.arabic-field textarea {
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Noto Sans Arabic', 'Arial', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.8 !important;
}

/* ── Labels ── */
.gradio-container label span {
    color: #111111 !important;
    font-size: 0.8rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* ── Accordion ── */
.gradio-container .accordion {
    background: #ffffff !important;
    border: 2px solid #111111 !important;
    border-radius: 0px !important;
    box-shadow: 3px 3px 0px #111111 !important;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 3px solid #111111;
    margin: 1.5rem 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #111111;
    font-weight: 800;
    font-size: 0.8rem;
    padding: 1.5rem 0;
    border-top: 3px solid #111111;
    margin-top: 2rem;
    text-transform: uppercase;
}

/* hide gradio footer */
footer { display: none !important; }
"""


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui():
    sample_products = load_sample_products()
    sample_choices = (
        ["── or paste your own reviews below ──"] +
        list(sample_products.keys())
    )

    with gr.Blocks(css=CSS, title="Moms Verdict") as demo:

        # ── Header ──
        gr.HTML("""
        <div class="app-header">
            <div class="app-title">🌸 Moms Verdict</div>
            <div class="app-subtitle">AI-powered review synthesis for Mumzworld</div>
            <div class="app-tagline">Reads every review. Writes the verdict moms actually need.</div>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── Left column — Input ──
            with gr.Column(scale=1, min_width=300):

                gr.HTML('<div class="section-label">Sample Products</div>')
                sample_dropdown = gr.Dropdown(
                    choices=sample_choices,
                    value=sample_choices[1],  # default to Avent bottle
                    label="Select a product",
                    container=True
                )

                gr.HTML('<hr class="divider">')

                with gr.Accordion("🔧 Custom input (paste your own reviews)", open=False):
                    gr.HTML('<div class="section-label">Product Name</div>')
                    custom_product = gr.Textbox(
                        placeholder="e.g. Chicco NaturalFit Bottle 250ml",
                        label="",
                        lines=1
                    )
                    gr.HTML('<div class="section-label" style="margin-top:0.75rem">Reviews JSON</div>')
                    custom_reviews = gr.Textbox(
                        placeholder='[{"text": "Great bottle!", "rating": 5.0, "language": "en"}, ...]',
                        label="",
                        lines=6
                    )

                gr.HTML('<hr class="divider">')

                run_btn = gr.Button(
                    "Generate Verdict →",
                    elem_classes=["generate-btn"]
                )

                gr.HTML("""
                <div style="margin-top: 1rem; color: #3a3a4a; font-size: 0.75rem; line-height: 1.6;">
                    Pipeline: Fake detection → Theme clustering →<br>
                    Structured extraction → Confidence scoring →<br>
                    Native Arabic generation
                </div>
                """)

            # ── Right column — Output ──
            with gr.Column(scale=2):

                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="section-label">Verdict — English</div>')
                        verdict_en = gr.Textbox(
                            label="",
                            lines=5,
                            interactive=False,
                            elem_classes=["output-textbox"]
                        )
                    with gr.Column():
                        gr.HTML('<div class="section-label">الحكم — Arabic</div>')
                        verdict_ar = gr.Textbox(
                            label="",
                            lines=5,
                            interactive=False,
                            elem_classes=["output-textbox", "arabic-field"]
                        )

                gr.HTML('<div class="section-label" style="margin-top:1rem">Pros & Cons — Grounded in Reviews</div>')
                pros_cons = gr.Textbox(
                    label="",
                    lines=12,
                    interactive=False,
                    elem_classes=["output-textbox"]
                )

                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="section-label">Confidence</div>')
                        confidence_out = gr.Textbox(
                            label="",
                            interactive=False,
                            elem_classes=["output-textbox"]
                        )
                    with gr.Column():
                        gr.HTML('<div class="section-label">Review Authenticity</div>')
                        fake_out = gr.Textbox(
                            label="",
                            interactive=False,
                            elem_classes=["output-textbox"]
                        )

                gr.HTML('<div class="section-label" style="margin-top:0.75rem">Themes Identified</div>')
                themes_out = gr.Textbox(
                    label="",
                    interactive=False,
                    elem_classes=["output-textbox"]
                )

                with gr.Accordion("{ } Raw JSON Output", open=False):
                    raw_json = gr.Code(
                        label="",
                        language="json",
                        interactive=False
                    )

        gr.HTML("""
        <div class="footer">
            Moms Verdict · Built for Mumzworld · 
            Powered by OpenRouter + sentence-transformers + Gradio
        </div>
        """)

        run_btn.click(
            fn=run_pipeline,
            inputs=[sample_dropdown, custom_reviews, custom_product],
            outputs=[
                verdict_en, verdict_ar, pros_cons,
                confidence_out, fake_out, themes_out, raw_json
            ]
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True)
