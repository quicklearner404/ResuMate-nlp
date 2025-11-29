import streamlit as st
import os
import sys
import time
import random

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

from src.parsing.parser import process_resume
from src.frontend.nlp_model import (
    ml_score,
    generate_recommendations,
    predict_job_title
)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ResuMate Platform",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CACHING ---
@st.cache_data(show_spinner=False)
def analyze_resume_fast(file):
    return process_resume(file)

# --- SESSION STATE ---
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

def trigger_analysis():
    st.session_state.analyzed = True

# --- LOADING TIPS ---
loading_tips = [
    "💡 Tip: ATS systems often reject resumes with complex columns or graphics.",
    "💡 Did you know? 75% of resumes are never seen by a human recruiter.",
    "💡 Tip: Use standard section headers like 'Experience' and 'Education'.",
    "💡 Fact: Tailoring your resume keywords increases interview chances by 50%.",
    "💡 Tip: Avoid putting your photo on US/UK resumes to prevent bias.",
    "💡 Fact: Most recruiters spend less than 7 seconds scanning a resume.",
    "💡 Tip: PDF is the safest format to ensure your formatting stays intact.",
    "💡 Tip: Quantify your achievements! (e.g., 'Increased sales by 20%')."
]

# --- CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .skill-chip {
        display: inline-flex; align-items: center; background-color: #f1f5f9; color: #334155; 
        border: 1px solid #cbd5e1; padding: 4px 12px; border-radius: 16px; margin: 4px; 
        font-size: 0.85rem; font-weight: 600;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff; border: 1px solid #f0f2f6; padding: 10px; 
        border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    div.stButton > button {
        background-color: #2563eb; color: white; border-radius: 6px; height: 3rem; font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col1, col2 = st.columns([6, 1])
with col1:
    st.title("ResuMate Analytics")
    st.caption("Professional Resume & Job Description Matching Engine")

# --- INPUT SECTION ---
with st.expander("📂 Upload Resume & Job Details", expanded=not st.session_state.analyzed):
    col_input_1, col_input_2 = st.columns([1, 1.5], gap="large")
    with col_input_1:
        st.subheader("1. Candidate")
        uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])
    with col_input_2:
        st.subheader("2. Target")
        job_description = st.text_area("Job Description", height=130, placeholder="Paste JD here...")

    analyze_btn = st.button("Run Compatibility Analysis ⚡", type="primary", use_container_width=True, on_click=trigger_analysis)

# --- RESULTS SECTION ---
if st.session_state.analyzed:
    if uploaded_file and job_description.strip():
        
        # 🎲 Select a random tip
        tip = random.choice(loading_tips)
        
        # 🎬 COOL LOADING UI
        with st.status("🚀 Launching AI Agents...", expanded=True) as status:
            c1, c2 = st.columns([1, 4])
            with c1:
                # Scanning GIF
                st.image("https://cdn.dribbble.com/users/107759/screenshots/2436386/media/cfd558d7c433390f7079234f9a12c45e.gif", width=80)
            with c2:
                st.markdown(f"**{tip}**") # Random Fact
            
            st.write("📄 Parsing PDF structure (Cached)...")
            resume_text, entities = analyze_resume_fast(uploaded_file)
            
            st.write("🧠 Identifying technical entities...")
            predicted_role = predict_job_title(resume_text)
            
            st.write("⚖️ Calculating vector similarity...")
            score = ml_score(resume_text, job_description)
            recs = generate_recommendations(resume_text, job_description)
            
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        st.divider()
        
        # 1. KPI ROW
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1: st.metric("Match Score", f"{score}%", delta="Confidence")
        with kpi2: st.metric("Detected Role", predicted_role)
        with kpi3: st.metric("Skills Found", len(set(entities.get("Skills", []))))
        with kpi4: st.metric("Contact Info", "Present" if entities.get("Email") else "Missing")
        
        # 2. TABS
        tab_skills, tab_gaps, tab_hist, tab_json = st.tabs(["🛠️ Skills", "⚠️ Gaps", "🏢 History", "🔍 Raw"])

        with tab_skills:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.write(f"**Name:** {entities.get('Name', ['N/A'])[0]}")
                st.write(f"**Email:** {entities.get('Email', ['N/A'])[0]}")
            with c2:
                skills = list(set(entities.get("Skills", [])))
                if skills:
                    st.markdown("".join([f'<span class="skill-chip">{s}</span>' for s in skills]), unsafe_allow_html=True)
                else:
                    st.warning("No skills extracted.")

        with tab_gaps:
            if recs:
                for rec in recs:
                    if "Great match" in rec: st.success(rec)
                    else: st.warning(rec.replace("⚠️", "").strip())
            else: st.info("No suggestions available.")

        with tab_hist:
            comps = entities.get("Company", [])
            desigs = entities.get("Designation", [])
            if comps or desigs:
                for i in range(max(len(comps), len(desigs))):
                    r = desigs[i] if i < len(desigs) else "Role N/A"
                    c = comps[i] if i < len(comps) else "Company N/A"
                    st.markdown(f"**{r}** | {c}")
                    st.divider()
            else:
                st.caption("No work history found.")
            for d in entities.get("Degree", []): st.markdown(f"🎓 {d}")

        with tab_json:
            st.json(entities)
            with st.expander("See extracted text"): st.text(resume_text)

    else:
        st.error("Please upload a resume and paste a job description.")
        st.session_state.analyzed = False