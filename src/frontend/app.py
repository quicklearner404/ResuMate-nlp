import streamlit as st
from parser import extract_text_from_pdf
from nlp_model import calculate_similarity


st.set_page_config(
    page_title="ResuMate – AI Resume Analyzer",
    page_icon="💼",
    layout="centered",
)

st.markdown("""
<style>
    /* Background and font */
    body {
        background-color: #f9fafc;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title */
    .title {
        text-align: center;
        font-size: 36px !important;
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 10px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #576574;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 0.6rem 1.4rem;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background-color: #43a047;
        transform: scale(1.05);
    }

    /* Textarea */
    textarea {
        border-radius: 8px !important;
        border: 1px solid #dcdde1 !important;
        font-size: 16px !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: bold;
        font-size: 16px;
        color: #2c3e50;
    }

    /* Score box */
    .score-box {
        text-align: center;
        background: linear-gradient(135deg, #4CAF50, #66bb6a);
        padding: 15px;
        border-radius: 12px;
        color: white;
        font-size: 22px;
        font-weight: 600;
        margin-top: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }

</style>
""", unsafe_allow_html=True)


st.markdown("<div class='title'>💼 ResuMate – AI Resume Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload your resume and paste a job description below to check compatibility!</div>",
            unsafe_allow_html=True)


uploaded_file = st.file_uploader("📄 Upload Resume (PDF only)", type=["pdf"])
job_description = st.text_area("📝 Paste Job Description", height=200, placeholder="Paste the job description here...")


if st.button("🚀 Analyze Resume"):
    if uploaded_file is not None and job_description.strip() != "":
        with st.spinner("🤖 Analyzing your resume and job description..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            similarity_score = calculate_similarity(resume_text, job_description)

        # --- DISPLAY RESULTS ---
        st.markdown(f"<div class='score-box'>✅ Compatibility Score: {similarity_score:.2f}%</div>",
                    unsafe_allow_html=True)

        with st.expander("🔍 View Extracted Resume Text"):
            st.write(resume_text)

        st.success("Analysis complete! 🎉")
    else:
        st.warning("⚠️ Please upload a resume and paste a job description.")
