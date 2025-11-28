import streamlit as st
from src.parsing.parser import process_resume
from src.frontend.nlp_model import (
    ml_score,
    generate_recommendations,
    predict_job_title
)

st.set_page_config(page_title="💼 ResuMate – AI Resume Analyzer", layout="centered")

st.title("💼 ResuMate – AI Resume Analyzer")
st.markdown("""
Welcome to **ResuMate**, your AI-powered resume evaluation assistant!
Upload a resume and paste a job description to check compatibility, extract
important information, and receive improvement recommendations.
""")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF only)", type=["pdf"])
job_description = st.text_area("📝 Paste Job Description", height=200)

if st.button("🚀 Analyze"):
    if uploaded_file is not None and job_description.strip() != "":
        with st.spinner("Processing your resume... Please wait ⏳"):

            try:

                resume_text, parsed_entities = process_resume(uploaded_file)

                ml_prediction = predict_job_title(resume_text)

                similarity_score = ml_score(resume_text, job_description)

                recommendation = generate_recommendations(resume_text, job_description)

                st.success(f"✅ Compatibility Score: **{similarity_score}%**")

                st.subheader("🤖 ML Predicted Job Title")
                st.info(f"**{ml_prediction}**")

                st.subheader("📌 Extracted Information (BERT)")
                st.json(parsed_entities)

                st.subheader("💡 Recommendations")
                st.write(recommendation)

                with st.expander("📄 View Resume Text"):
                    st.write(resume_text)

            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")

    else:
        st.warning("⚠️ Please upload a resume and paste a job description.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit, BERT, ML models, and Sentence Transformers.")
