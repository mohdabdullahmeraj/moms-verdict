"""
app.py

Gradio UI for Moms Verdict pipeline.
Run with: python app.py
Colab: demo.launch(share=True) gives a public URL.
"""

import json
import os
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import MomsVerdictPipeline
from src.schema import ConfidenceLevel

# initialise pipeline once — embedding model loads lazily on first use
pipeline = MomsVerdictPipeline()

SAMPLE_DIR = Path("data/sample_products")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sample_products():
    """Load available sample products from index.json."""
    index_path = SAMPLE_DIR / "index.json"
    if not index_path.exists():
        return {}
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    return {item["product_name"]: item["file"] for item in index}


def format_pros_cons(pros, cons):
    """Format pros and cons into readable text."""
    lines = []
    if pros:
        lines.append("✅ PROS")
        for p in pros:
            lines.append(
                f"  • {p.point}\n"
                f"    ({p.evidence.mention_count} moms, "
                f"{p.mention_percentage:.0f}%)\n"
                f"    \"{p.evidence.representative_quote}\""
            )
    if cons:
        lines.append("\n❌ CONS")
        for c in cons:
            lines.append(
                f"  • {c.point}\n"
                f"    ({c.evidence.mention_count} moms, "
                f"{c.mention_percentage:.0f}%)\n"
                f"    \"{c.evidence.representative_quote}\""
            )
    return "\n".join(lines) if lines else "No pros/cons extracted."


def confidence_color(level):
    colors = {
        "high": "🟢 HIGH",
        "medium": "🟡 MEDIUM",
        "low": "🔴 LOW",
        "insufficient": "⛔ INSUFFICIENT"
    }
    return colors.get(level, level)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_pipeline(product_name, reviews_json, sample_choice):
    """
    Called when the user clicks Generate Verdict.
    Either uses the typed JSON or loads a sample product.
    """
    try:
        # decide input source
        if sample_choice and sample_choice != "-- Type your own --":
            products = load_sample_products()
            filename = products.get(sample_choice)
            if filename:
                filepath = SAMPLE_DIR / filename
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                product_name = data["product_name"]
                raw_reviews = data["reviews"]
            else:
                return ("Error: sample file not found.",) + ("",) * 6
        elif reviews_json.strip():
            try:
                raw_reviews = json.loads(reviews_json)
                if not isinstance(raw_reviews, list):
                    return ("Error: reviews must be a JSON array.",) + ("",) * 6
            except json.JSONDecodeError as e:
                return (f"Error parsing JSON: {e}",) + ("",) * 6
        else:
            return ("Please select a sample product or paste reviews JSON.",) + ("",) * 6

        if not product_name.strip():
            return ("Please enter a product name.",) + ("",) * 6

        # run the pipeline
        verdict = pipeline.run(product_name.strip(), raw_reviews)

        # format outputs
        if verdict.confidence_level == ConfidenceLevel.INSUFFICIENT:
            verdict_en_out = f"⛔ Verdict refused\n\n{verdict.refusal_reason}"
            verdict_ar_out = ""
            pros_cons_out = ""
        else:
            verdict_en_out = verdict.verdict_en
            verdict_ar_out = verdict.verdict_ar
            pros_cons_out = format_pros_cons(verdict.pros, verdict.cons)

        confidence_out = (
            f"{confidence_color(verdict.confidence_level.value)} — "
            f"Score: {verdict.confidence_score} | "
            f"Reviews: {verdict.review_count} | "
            f"Languages: {verdict.language_breakdown}"
        )

        fake_out = (
            f"⚠️ FLAGGED — {verdict.fake_review_flag.reason}"
            if verdict.fake_review_flag.flagged
            else f"✅ Reviews appear genuine "
                 f"(similarity score: "
                 f"{verdict.fake_review_flag.average_similarity_score:.3f})"
        )

        themes_out = (
            ", ".join(verdict.themes_identified)
            if verdict.themes_identified
            else "None identified"
        )

        raw_out = verdict.model_dump_json(indent=2)

        return (
            verdict_en_out,
            verdict_ar_out,
            pros_cons_out,
            confidence_out,
            fake_out,
            themes_out,
            raw_out
        )

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return (
            f"Pipeline error: {e}",
            "", "", "", "", "",
            error_detail
        )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    sample_products = load_sample_products()
    sample_choices = ["-- Type your own --"] + list(sample_products.keys())

    with gr.Blocks(
        title="Moms Verdict — Mumzworld Review Intelligence",
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown("""
        # 🌸 Moms Verdict
        ### AI-powered review synthesis for Mumzworld
        *Reads every review. Writes the verdict moms actually need.*
        ---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")

                sample_dropdown = gr.Dropdown(
                    choices=sample_choices,
                    value=sample_choices[0],
                    label="Load a sample product",
                    info="Select a pre-loaded product or type your own below"
                )

                product_input = gr.Textbox(
                    label="Product Name",
                    placeholder="e.g. Philips Avent Natural Baby Bottle 260ml",
                    lines=1
                )

                reviews_input = gr.Textbox(
                    label="Reviews JSON (if not using sample)",
                    placeholder='[{"text": "Great bottle!", "rating": 5.0, "language": "en"}, ...]',
                    lines=8,
                    info="Paste a JSON array of reviews. Each needs: text, rating, language."
                )

                run_btn = gr.Button(
                    "🔍 Generate Verdict",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=2):
                gr.Markdown("### Output")

                with gr.Row():
                    verdict_en = gr.Textbox(
                        label="Verdict (English)",
                        lines=4,
                        interactive=False
                    )
                    verdict_ar = gr.Textbox(
                        label="الحكم (Arabic)",
                        lines=4,
                        interactive=False,
                        rtl=True
                    )

                pros_cons = gr.Textbox(
                    label="Pros & Cons (Grounded)",
                    lines=10,
                    interactive=False
                )

                confidence_out = gr.Textbox(
                    label="Confidence",
                    interactive=False
                )

                fake_out = gr.Textbox(
                    label="Review Authenticity",
                    interactive=False
                )

                themes_out = gr.Textbox(
                    label="Themes Identified",
                    interactive=False
                )

        with gr.Accordion("Raw JSON Output", open=False):
            raw_json = gr.Code(
                label="Full MomsVerdict JSON",
                language="json",
                interactive=False
            )

        gr.Markdown("""
        ---
        **How it works:** Reviews → Fake detection → Theme clustering →
        Structured extraction → Confidence scoring → Native Arabic generation
        """)

        run_btn.click(
            fn=run_pipeline,
            inputs=[product_input, reviews_input, sample_dropdown],
            outputs=[
                verdict_en, verdict_ar, pros_cons,
                confidence_out, fake_out, themes_out, raw_json
            ]
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    # share=True gives a public URL in Colab
    demo.launch(share=True)
