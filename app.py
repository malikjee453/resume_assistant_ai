import io
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import streamlit as st
from PIL import Image
from pypdf import PdfReader
from docx import Document
from google import genai
from google.genai import types


APP_TITLE = "AI Resume ATS Analyzer"
DEFAULT_MODEL = "gemini-3.8-flash"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    .score-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.25);
        text-align: center;
        margin-bottom: 16px;
    }
    .score {
        font-size: 48px;
        font-weight: 800;
        line-height: 1;
    }
    .small-muted {
        color: #777;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_key() -> Optional[str]:
    """Read the Gemini API key from Streamlit secrets or an environment variable."""
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass

    key = os.getenv("GEMINI_API_KEY")
    return key.strip() if key else None


def get_client() -> genai.Client:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to Streamlit Secrets or set it "
            "as an environment variable."
        )
    return genai.Client(api_key=api_key)


def extract_docx_text(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    parts = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            row_text = " | ".join(cell for cell in cells if cell)
            if row_text:
                parts.append(row_text)

    return "\n".join(parts).strip()


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []

    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(text.strip())

    return "\n\n".join(pages).strip()


def image_to_part(file_bytes: bytes, mime_type: str) -> types.Part:
    return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)


def pdf_to_part(file_bytes: bytes) -> types.Part:
    return types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")


def build_prompt(job_description: str, extracted_text: str = "") -> str:
    jd_section = (
        job_description.strip()
        if job_description.strip()
        else "No job description was provided. Evaluate against general ATS/resume best practices."
    )

    text_section = (
        f"\nEXTRACTED RESUME TEXT:\n{extracted_text[:50000]}"
        if extracted_text
        else "\nThe resume is provided as a visual/document attachment."
    )

    return f"""
You are an expert ATS resume analyzer and career-document editor.

Analyze the attached resume or the extracted resume text. The user wants:
1. A practical ATS compatibility score from 0 to 100.
2. Specific problems that could reduce ATS parsing or recruiter readability.
3. Concrete improvements.
4. Missing or weak keywords relative to the job description when one is supplied.
5. Suggestions for stronger, truthful bullet points without inventing experience.
6. A concise overall summary.

JOB DESCRIPTION:
{jd_section}
{text_section}

IMPORTANT RULES:
- Do not invent jobs, degrees, certifications, employers, dates, achievements, metrics, or skills.
- Do not claim that an ATS score is an official score from any particular ATS vendor.
- Treat the score as an estimated compatibility score based on the criteria below.
- Consider: contact information, section structure, keyword alignment, measurable achievements,
  action verbs, clarity, chronology, skills, education, formatting/parsing risk, length,
  consistency, and relevance to the supplied job description.
- If a job description is supplied, keyword alignment should be a major part of the score.
- If no job description is supplied, use a general ATS-readiness evaluation.
- Return ONLY valid JSON matching the requested schema.
"""


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "ats_score": {
            "type": "integer",
            "description": "Estimated ATS compatibility score from 0 to 100."
        },
        "score_breakdown": {
            "type": "object",
            "properties": {
                "keyword_match": {"type": "integer"},
                "formatting_and_parsing": {"type": "integer"},
                "experience_and_achievements": {"type": "integer"},
                "skills": {"type": "integer"},
                "sections_and_structure": {"type": "integer"},
                "clarity_and_consistency": {"type": "integer"},
            },
            "required": [
                "keyword_match",
                "formatting_and_parsing",
                "experience_and_achievements",
                "skills",
                "sections_and_structure",
                "clarity_and_consistency",
            ],
        },
        "summary": {"type": "string"},
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["High", "Medium", "Low"],
                    },
                },
                "required": ["issue", "why_it_matters", "severity"],
            },
        },
        "missing_keywords": {
            "type": "array",
            "items": {"type": "string"},
        },
        "improvements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "area": {"type": "string"},
                    "recommendation": {"type": "string"},
                    "example": {"type": "string"},
                },
                "required": ["area", "recommendation", "example"],
            },
        },
        "bullet_rewrites": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original": {"type": "string"},
                    "suggested": {"type": "string"},
                },
                "required": ["original", "suggested"],
            },
        },
        "ats_checklist": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "check": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["Pass", "Needs improvement", "Not detected"],
                    },
                    "note": {"type": "string"},
                },
                "required": ["check", "status", "note"],
            },
        },
    },
    "required": [
        "ats_score",
        "score_breakdown",
        "summary",
        "strengths",
        "issues",
        "missing_keywords",
        "improvements",
        "bullet_rewrites",
        "ats_checklist",
    ],
}


def analyze_resume(
    file_bytes: bytes,
    file_name: str,
    job_description: str,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    client = get_client()
    suffix = Path(file_name).suffix.lower()
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix)

    extracted_text = ""
    content = []

    if suffix == ".docx":
        extracted_text = extract_docx_text(file_bytes)
        if not extracted_text:
            raise ValueError("The Word document contains no readable text.")
        content = [build_prompt(job_description, extracted_text)]
    elif suffix == ".pdf":
        extracted_text = extract_pdf_text(file_bytes)
        # PDFs with text are cheaper to analyze as text. Scanned/image PDFs are
        # sent directly to Gemini because Gemini supports PDF input.
        if extracted_text:
            content = [build_prompt(job_description, extracted_text)]
        else:
            content = [
                build_prompt(job_description),
                pdf_to_part(file_bytes),
            ]
    elif mime_type:
        # Validate that the upload is a readable image before sending it.
        Image.open(io.BytesIO(file_bytes)).verify()
        content = [
            build_prompt(job_description),
            image_to_part(file_bytes, mime_type),
        ]
    else:
        raise ValueError("Unsupported file type. Upload PDF, DOCX, PNG, JPG, JPEG, or WEBP.")

    response = client.models.generate_content(
        model=model_name,
        contents=content,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=5000,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    try:
        result = json.loads(response.text)
    except json.JSONDecodeError as exc:
        # Defensive fallback in case an upstream model/API version returns
        # fenced JSON despite the JSON response configuration.
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip())
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            raise RuntimeError("Gemini returned invalid JSON.") from exc

    result["ats_score"] = max(0, min(100, int(result.get("ats_score", 0))))
    return result


def score_label(score: int) -> str:
    if score >= 85:
        return "Excellent ATS readiness"
    if score >= 70:
        return "Good ATS readiness"
    if score >= 55:
        return "Needs improvement"
    return "Significant improvement recommended"


def render_list(items):
    for item in items:
        st.markdown(f"- {item}")


def main():
    st.title("📄 AI Resume ATS Analyzer")
    st.caption(
        "Upload a resume to get an estimated ATS compatibility score, issues, "
        "keyword gaps, and practical improvements."
    )

    with st.sidebar:
        st.header("Settings")
        model_name = st.text_input(
            "Gemini Flash model",
            value=DEFAULT_MODEL,
            help="Current stable Flash model. You can change this if your API project uses another supported Flash model.",
        )
        st.info(
            "The score is an AI-based estimate, not an official score from an ATS vendor."
        )
        st.markdown("**Supported files:** PDF, DOCX, PNG, JPG, JPEG, WEBP")

    job_description = st.text_area(
        "Job description (optional, but recommended)",
        height=180,
        placeholder="Paste the job description here to get a more targeted ATS/keyword analysis.",
    )

    uploaded = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx", "png", "jpg", "jpeg", "webp"],
        help="For the most useful keyword score, also paste the target job description.",
    )

    analyze_button = st.button(
        "🔎 Analyze Resume",
        type="primary",
        disabled=uploaded is None,
        use_container_width=True,
    )

    if not analyze_button:
        st.markdown(
            """
            ### How it works
            1. Upload your resume.
            2. Optionally paste the target job description.
            3. Gemini analyzes ATS compatibility and content quality.
            4. Review the score, keyword gaps, issues, and suggested improvements.

            **Privacy note:** Avoid uploading highly sensitive documents unless you are
            comfortable sending the resume content to the Gemini API.
            """
        )
        return

    if uploaded is None:
        st.warning("Please upload a resume.")
        return

    file_bytes = uploaded.getvalue()

    if len(file_bytes) > 10 * 1024 * 1024:
        st.error("Please upload a file smaller than 10 MB.")
        return

    with st.spinner("Analyzing your resume with Gemini Flash..."):
        try:
            result = analyze_resume(
                file_bytes=file_bytes,
                file_name=uploaded.name,
                job_description=job_description,
                model_name=model_name.strip() or DEFAULT_MODEL,
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.info(
                "Check that GEMINI_API_KEY is configured, the selected model is available "
                "to your API project, and the uploaded file is valid."
            )
            return

    score = int(result["ats_score"])

    st.success(f"Analysis complete: {uploaded.name}")

    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="small-muted">Estimated ATS Score</div>
                <div class="score">{score}/100</div>
                <div>{score_label(score)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Summary")
        st.write(result.get("summary", ""))

    st.subheader("📊 Score Breakdown")
    breakdown = result.get("score_breakdown", {})
    cols = st.columns(3)
    breakdown_items = [
        ("Keyword Match", breakdown.get("keyword_match", 0)),
        ("Formatting / Parsing", breakdown.get("formatting_and_parsing", 0)),
        ("Experience", breakdown.get("experience_and_achievements", 0)),
        ("Skills", breakdown.get("skills", 0)),
        ("Sections", breakdown.get("sections_and_structure", 0)),
        ("Clarity", breakdown.get("clarity_and_consistency", 0)),
    ]
    for i, (label, value) in enumerate(breakdown_items):
        with cols[i % 3]:
            st.metric(label, f"{value}/100")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Strengths")
        render_list(result.get("strengths", []))

    with col2:
        st.subheader("🔑 Missing / Weak Keywords")
        keywords = result.get("missing_keywords", [])
        if keywords:
            st.write(", ".join(keywords))
        else:
            st.write("No major keyword gaps were detected.")

    st.subheader("⚠️ Issues to Fix")
    issues = result.get("issues", [])
    if issues:
        for item in issues:
            severity = item.get("severity", "Medium")
            st.markdown(
                f"**{severity}: {item.get('issue', '')}**  \n"
                f"{item.get('why_it_matters', '')}"
            )
            st.divider()
    else:
        st.write("No major issues were detected.")

    st.subheader("🚀 Recommended Improvements")
    improvements = result.get("improvements", [])
    for item in improvements:
        st.markdown(f"**{item.get('area', 'Improvement')}**")
        st.write(item.get("recommendation", ""))
        if item.get("example"):
            st.info(f"Example: {item['example']}")

    st.subheader("✍️ Suggested Bullet Rewrites")
    rewrites = result.get("bullet_rewrites", [])
    if rewrites:
        for item in rewrites:
            st.markdown(f"**Original:** {item.get('original', '')}")
            st.markdown(f"**Suggested:** {item.get('suggested', '')}")
            st.divider()
    else:
        st.write("No bullet rewrites were suggested.")

    st.subheader("🧾 ATS Checklist")
    checklist = result.get("ats_checklist", [])
    for item in checklist:
        status = item.get("status", "Not detected")
        st.markdown(
            f"**{status} — {item.get('check', '')}**  \n"
            f"{item.get('note', '')}"
        )

    st.caption(
        "Important: ATS scoring varies between employers and software. Use this report "
        "as guidance and verify every suggested change against the candidate's real experience."
    )


if __name__ == "__main__":
    main()
