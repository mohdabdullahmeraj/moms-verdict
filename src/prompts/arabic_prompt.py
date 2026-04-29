"""
src/prompts/arabic_prompt.py

The Arabic generation prompt.

Critical design decisions documented here:

1. THE PROMPT IS IN ARABIC.
   This is intentional. Prompting in Arabic activates the model's
   Arabic language capabilities more directly. An English instruction
   saying "write in Arabic" produces worse Arabic than an Arabic
   instruction saying the same thing. Empirically verified.

2. WE PROVIDE FACTS, NOT THE ENGLISH VERDICT.
   The model receives structured data — pros list, cons list, sentiment,
   count. It does NOT receive verdict_en. This ensures verdict_ar is
   independently generated from the same data source, not a translation.

3. GULF REGISTER SPECIFICATION.
   The prompt explicitly asks for "العربية الفصحى المبسطة المناسبة
   للقارئة الخليجية" — simplified Modern Standard Arabic appropriate
   for a Gulf reader. This is the right register for a product page
   targeting GCC mothers: formal enough to be trustworthy, warm enough
   to feel human.

4. EXPLICIT ANTI-TRANSLATION INSTRUCTION.
   "لا تترجمي من الإنجليزية" — do not translate from English.
   This is a direct instruction because without it, models sometimes
   default to translation-style output even when given structured data.

5. LENGTH CONSTRAINT IN THE PROMPT.
   "جملتان أو ثلاث جمل كحد أقصى" — two or three sentences maximum.
   Product page copy needs to be scannable. Long Arabic paragraphs
   defeat the purpose.
"""

from src.schema import ProConItem, ConfidenceLevel, OverallSentiment


# ---------------------------------------------------------------------------
# Sentiment mapping to Arabic
# ---------------------------------------------------------------------------

SENTIMENT_AR = {
    OverallSentiment.POSITIVE: "إيجابي بشكل عام",
    OverallSentiment.MIXED: "متباين بين الإيجابي والسلبي",
    OverallSentiment.NEGATIVE: "سلبي بشكل عام"
}

CONFIDENCE_AR = {
    ConfidenceLevel.HIGH: "عالية (بناءً على عدد كبير من التقييمات)",
    ConfidenceLevel.MEDIUM: "متوسطة (بناءً على عدد معقول من التقييمات)",
    ConfidenceLevel.LOW: "منخفضة (عدد محدود من التقييمات)",
    ConfidenceLevel.INSUFFICIENT: "غير كافية"
}


# ---------------------------------------------------------------------------
# Prompt template (written in Arabic)
# ---------------------------------------------------------------------------

ARABIC_PROMPT_TEMPLATE = """\
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
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_arabic_prompt(
    product_name: str,
    pros: list[ProConItem],
    cons: list[ProConItem],
    overall_sentiment: OverallSentiment,
    confidence_level: ConfidenceLevel,
    review_count: int,
    language_breakdown: dict[str, int]
) -> str:
    """
    Build the Arabic generation prompt from structured extraction data.

    Converts Python objects into formatted Arabic text sections
    that the model can reason about directly.

    Args:
        product_name: Product name (English is fine — kept as-is)
        pros: List of validated ProConItem objects
        cons: List of validated ProConItem objects
        overall_sentiment: OverallSentiment enum value
        confidence_level: ConfidenceLevel enum value
        review_count: Total reviews processed
        language_breakdown: e.g. {"en": 98, "ar": 35}

    Returns:
        Complete Arabic prompt string
    """

    # format pros as Arabic bullet list
    if pros:
        pros_lines = []
        for i, pro in enumerate(pros, start=1):
            pros_lines.append(
                f"  {i}. {pro.point} "
                f"(مذكور في {pro.evidence.mention_count} تقييم، "
                f"{pro.mention_percentage:.0f}٪ من الإجمالي)"
            )
        pros_ar = "\n".join(pros_lines)
    else:
        pros_ar = "  لم تُذكر إيجابيات محددة في هذه التقييمات."

    # format cons as Arabic bullet list
    if cons:
        cons_lines = []
        for i, con in enumerate(cons, start=1):
            cons_lines.append(
                f"  {i}. {con.point} "
                f"(مذكور في {con.evidence.mention_count} تقييم، "
                f"{con.mention_percentage:.0f}٪ من الإجمالي)"
            )
        cons_ar = "\n".join(cons_lines)
    else:
        cons_ar = "  لم تُذكر سلبيات محددة في هذه التقييمات."

    # format language breakdown
    lang_parts = []
    if language_breakdown.get("en", 0) > 0:
        lang_parts.append(f"{language_breakdown['en']} تقييم بالإنجليزية")
    if language_breakdown.get("ar", 0) > 0:
        lang_parts.append(f"{language_breakdown['ar']} تقييم بالعربية")
    language_breakdown_ar = "، ".join(lang_parts) if lang_parts else "غير محدد"

    return ARABIC_PROMPT_TEMPLATE.format(
        product_name=product_name,
        review_count=review_count,
        sentiment_ar=SENTIMENT_AR.get(overall_sentiment, "غير محدد"),
        confidence_ar=CONFIDENCE_AR.get(confidence_level, "غير محدد"),
        language_breakdown_ar=language_breakdown_ar,
        pros_ar=pros_ar,
        cons_ar=cons_ar
    )
