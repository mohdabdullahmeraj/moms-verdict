import gradio as gr
import json
from pathlib import Path
from src.pipeline import MomsVerdictPipeline

# Initialize the pipeline once globally
print("Initializing pipeline...")
pipeline = MomsVerdictPipeline()
print("Pipeline ready.")

def get_available_products():
    path = Path("data/sample_products")
    if not path.exists():
        return []
    return [f.name for f in path.glob("*.json")]

def format_pro_con(items):
    if not items:
        return "*None reported.*"
    
    lines = []
    for item in items:
        lines.append(f"**{item.point}** ({item.mention_percentage:.1f}%)")
        lines.append(f"> *\"{item.evidence.representative_quote}\"*")
        lines.append("")
    return "\n".join(lines)

def run_analysis(product_file):
    if not product_file:
        return "Please select a product from the dropdown.", "", "", "", "", "", ""
        
    file_path = Path("data/sample_products") / product_file
    
    try:
        verdict = pipeline.run_from_file(str(file_path))
        
        # Build UI components
        
        # Verdicts
        en_verdict = verdict.verdict_en if verdict.verdict_en else f"*Insufficient Data:* {verdict.refusal_reason}"
        ar_verdict = f"<div dir='rtl' style='text-align: right; font-size: 1.1em;'>{verdict.verdict_ar}</div>" if verdict.verdict_ar else "<div dir='rtl' style='text-align: right;'>*غير متوفر (بيانات غير كافية)*</div>"
        
        # Metrics
        sentiment = f"**{verdict.overall_sentiment.value.upper()}**"
        confidence = f"**{verdict.confidence_level.value.upper()}** ({verdict.confidence_score})"
        
        if verdict.fake_review_flag.flagged:
            fake_flag = f"🚨 **FLAGGED** (Similarity: {verdict.fake_review_flag.average_similarity_score:.2f})"
        else:
            fake_flag = "✅ **CLEAN**"
            
        metrics_md = f"""
| Sentiment | Confidence | Fake Review Check |
| :---: | :---: | :---: |
| {sentiment} | {confidence} | {fake_flag} |
"""
        
        # Pros and Cons
        pros_md = format_pro_con(verdict.pros)
        cons_md = format_pro_con(verdict.cons)
        
        # Themes
        themes = ", ".join(verdict.themes_identified) if verdict.themes_identified else "None"
        themes_md = f"**Themes Identified:** {themes}"
        
        # Raw JSON for accordion
        raw_json = verdict.model_dump_json(indent=2)
        
        return metrics_md, en_verdict, ar_verdict, pros_md, cons_md, themes_md, raw_json
        
    except Exception as e:
        return f"Error: {str(e)}", "", "", "", "", "", ""

# Build Gradio UI
with gr.Blocks(title="Moms Verdict Pipeline", theme=gr.themes.Soft(primary_hue="rose")) as app:
    gr.Markdown("# 👩‍🍼 Moms Verdict AI Pipeline")
    gr.Markdown("Select a product to run the end-to-end review analysis pipeline. The pipeline will clean data, detect fake reviews, cluster by theme, extract structured pros/cons via LLM, validate facts, and generate a native Arabic verdict.")
    
    with gr.Row():
        product_dropdown = gr.Dropdown(
            choices=get_available_products(),
            label="Select Product Data",
            info="Loads JSON files from data/sample_products/"
        )
        run_btn = gr.Button("Run Pipeline", variant="primary")
        
    gr.Markdown("---")
    
    metrics_out = gr.Markdown()
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🇬🇧 English Verdict")
            en_verdict_out = gr.Markdown()
        with gr.Column():
            gr.Markdown("<h3 dir='rtl' style='text-align: right;'>الخلاصة بالعربية 🇸🇦</h3>")
            ar_verdict_out = gr.HTML()
            
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ✅ Top Pros (Grounded)")
            pros_out = gr.Markdown()
        with gr.Column():
            gr.Markdown("### ❌ Top Cons (Grounded)")
            cons_out = gr.Markdown()
            
    themes_out = gr.Markdown()
    
    with gr.Accordion("View Raw JSON Output", open=False):
        raw_json_out = gr.Code(language="json")
        
    # Wire up the button
    run_btn.click(
        fn=run_analysis,
        inputs=[product_dropdown],
        outputs=[metrics_out, en_verdict_out, ar_verdict_out, pros_out, cons_out, themes_out, raw_json_out]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
