"""ResuMate Streamlit app: upload a resume, paste a job description, see the fit.

    streamlit run app/streamlit_app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.append(str(APP_DIR.parent))

from resumate.extraction.pdf_parser import process_resume  # noqa: E402
from resumate.matching.role_classifier import (  # noqa: E402
    ml_score,
    predict_job_title,
    skill_gap,
)

SAMPLE_RESUME = APP_DIR / "samples" / "resume.pdf"
SAMPLE_JOB = APP_DIR / "samples" / "job_description.txt"

st.set_page_config(page_title="ResuMate", page_icon="📄", layout="wide")

st.markdown(
    """
<style>
  .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
  h1 { margin-bottom: 0 !important; }
  .subtitle { color: #64748b; margin-top: 0.2rem; margin-bottom: 1.2rem; }
  .pill { display: inline-block; padding: 2px 10px; margin: 0 6px 6px 0; border-radius: 999px;
          font-size: 0.78rem; background: #eef2ff; color: #3730a3; border: 1px solid #e0e7ff; }
  .chip { display: inline-block; padding: 4px 12px; margin: 0 6px 8px 0; border-radius: 999px;
          font-size: 0.85rem; font-weight: 600; border: 1px solid; }
  .chip-ok   { background: #ecfdf5; color: #047857; border-color: #a7f3d0; }
  .chip-miss { background: #fffbeb; color: #b45309; border-color: #fde68a; }
  .chip-misc { background: #f8fafc; color: #334155; border-color: #e2e8f0; }
  .field { color: #64748b; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .04em;
           margin-top: 0.6rem; }
  .value { font-size: 1rem; color: #0f172a; }
  .muted { color: #94a3b8; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def analyze(pdf_bytes: bytes, job_description: str):
    import io

    resume_text, fields = process_resume(io.BytesIO(pdf_bytes))
    matched, missing = skill_gap(resume_text, job_description)
    return {
        "text": resume_text,
        "fields": fields,
        "score": ml_score(resume_text, job_description),
        # Prefer the job title the model extracted; fall back to a keyword guess.
        "role": (fields.get("Designation") or [predict_job_title(resume_text)])[0],
        "matched": matched,
        "missing": missing,
    }


def chips(items, kind):
    if not items:
        return '<span class="muted">None</span>'
    return "".join(f'<span class="chip chip-{kind}">{i}</span>' for i in items)


def unique(values):
    """Drop case-insensitive duplicates, keeping the first spelling."""
    seen, out = set(), []
    for v in values:
        if v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def field(label, values, limit=1):
    values = unique(values)
    shown = ", ".join(values[:limit]) if values else '<span class="muted">Not found</span>'
    st.markdown(f'<div class="field">{label}</div><div class="value">{shown}</div>', unsafe_allow_html=True)


def fit_label(score):
    if score >= 60:
        return "Strong match"
    if score >= 35:
        return "Moderate match"
    return "Low match"


# --------------------------------------------------------------------------- #
# Header and inputs
# --------------------------------------------------------------------------- #
st.title("📄 ResuMate")
st.markdown(
    '<div class="subtitle">Extract structured fields from a PDF resume and check how well it fits a job.</div>'
    '<span class="pill">DistilBERT NER</span><span class="pill">Rule-based email, phone, skills</span>'
    '<span class="pill">TF-IDF job matching</span>',
    unsafe_allow_html=True,
)

if "use_sample" not in st.session_state:
    st.session_state.use_sample = False

with st.container(border=True):
    col_resume, col_job = st.columns([1, 1.4], gap="large")
    with col_resume:
        st.markdown("**1. Resume**")
        uploaded = st.file_uploader("PDF resume", type=["pdf"], label_visibility="collapsed")
        if st.session_state.use_sample and not uploaded:
            st.caption(f"Using sample resume: `{SAMPLE_RESUME.name}` (fictional candidate)")
    with col_job:
        st.markdown("**2. Job description**")
        default_job = SAMPLE_JOB.read_text(encoding="utf-8") if st.session_state.use_sample else ""
        job_description = st.text_area(
            "Job description", value=default_job, height=150,
            placeholder="Paste the job description here...", label_visibility="collapsed",
        )

    b1, b2, _ = st.columns([1.2, 1.2, 3])
    run = b1.button("Analyze resume", type="primary", use_container_width=True)
    if b2.button("Try the sample", use_container_width=True):
        st.session_state.use_sample = True
        st.rerun()

pdf_bytes = uploaded.getvalue() if uploaded else (
    SAMPLE_RESUME.read_bytes() if st.session_state.use_sample else None
)

if not (run or st.session_state.use_sample):
    st.info("Upload a PDF resume and paste a job description, or click **Try the sample**.")
    st.stop()
if not pdf_bytes or not job_description.strip():
    st.warning("Please provide both a resume and a job description.")
    st.stop()

with st.spinner("Reading the resume and running the model..."):
    result = analyze(pdf_bytes, job_description)

fields = result["fields"]
if "Error" in fields:
    st.error("Could not read any text from this PDF.")
    st.stop()

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
required = len(result["matched"]) + len(result["missing"])
k1, k2, k3, k4 = st.columns(4)
with k1, st.container(border=True):
    st.metric("Match score", f"{result['score']:.0f}%")
    st.progress(min(int(result["score"]), 100), text=fit_label(result["score"]))
with k2, st.container(border=True):
    st.metric("Job skills covered", f"{len(result['matched'])} / {required}" if required else "–")
with k3, st.container(border=True):
    st.metric("Skills on resume", len(set(fields.get("Skills", []))))
with k4, st.container(border=True):
    st.markdown(
        f'<div style="font-size:0.875rem;color:#31333f">Current role</div>'
        f'<div style="font-size:1.35rem;line-height:1.3;margin-top:0.35rem">{result["role"]}</div>',
        unsafe_allow_html=True,
    )

left, right = st.columns([1, 1.4], gap="large")

with left, st.container(border=True):
    st.subheader("Candidate")
    field("Name", fields.get("Name", []))
    field("Email", fields.get("Email", []))
    field("Phone", fields.get("Phone", []))
    field("Location", fields.get("Location", []))
    field("Roles", fields.get("Designation", []), limit=3)
    field("Companies", fields.get("Company", []), limit=3)
    field("Education", fields.get("Degree", []) + fields.get("College", []), limit=2)

with right, st.container(border=True):
    st.subheader("Fit for this job")
    st.markdown('<div class="field">Required skills found on the resume</div>', unsafe_allow_html=True)
    st.markdown(chips(result["matched"], "ok"), unsafe_allow_html=True)
    st.markdown('<div class="field">Required skills missing</div>', unsafe_allow_html=True)
    st.markdown(chips(result["missing"], "miss"), unsafe_allow_html=True)
    other = sorted(set(fields.get("Skills", [])) - set(result["matched"]))
    st.markdown('<div class="field">Other skills on the resume</div>', unsafe_allow_html=True)
    st.markdown(chips(other, "misc"), unsafe_allow_html=True)
    if result["missing"]:
        st.caption("Tip: if you have the missing skills, mention them explicitly. Matching is keyword-based.")

with st.expander("Raw extraction"):
    c1, c2 = st.columns(2)
    c1.json(fields)
    c2.text(result["text"])

st.caption(os.environ.get("RESUMATE_FOOTER", "Model: quicklearner404/DistillbertNER · see README for evaluation results"))
