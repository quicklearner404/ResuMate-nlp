import streamlit as st
from src.parsing.parser import process_resume
# from Fatima’s parser
from src.frontend.nlp_model import calculate_similarity, generate_recommendations

# --- PAGE SETUP ---
st.set_page_config(page_title="💼 ResuMate – AI Resume Analyzer", layout="centered")

# --- HEADER ---
st.title("💼 ResuMate – AI Resume Analyzer")
st.markdown("""
Welcome to **ResuMate**, your AI-powered career assistant!
Upload your resume and paste a job description to check how well your profile matches.
""")

# --- INPUTS ---
uploaded_file = st.file_uploader("📄 Upload Resume (PDF only)", type=["pdf"])
job_description = st.text_area("📝 Paste Job Description", height=200)

# --- PROCESSING ---
if st.button("🚀 Analyze"):
    if uploaded_file is not None and job_description.strip() != "":
        with st.spinner("Processing your resume... Please wait ⏳"):
            try:
                # --- Step 1: Parse resume (Fatima’s part)
                resume_text, parsed_entities = process_resume(uploaded_file)

                # --- Step 2: Calculate similarity (Your part)
                similarity_score = calculate_similarity(resume_text, job_description)
                recommendation = generate_recommendations(resume_text, job_description)

                # --- Step 3: Display results ---
                st.success(f"✅ Compatibility Score: **{similarity_score:.2f}%**")

                # --- Parsed Entities ---
                st.subheader("📊 Extracted Information")
                st.json(parsed_entities)
                st.subheader("Recommendation to improve your CV")
                st.write(recommendation)
                # --- Raw Resume Text ---
                with st.expander("🧾 View Extracted Resume Text"):
                    st.write(resume_text)

            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
    else:
        st.warning("⚠️ Please upload a resume and paste a job description before analyzing.")

# --- FOOTER ---
st.markdown("---")
st.caption("Made with ❤️ using Streamlit, Sentence Transformers, and spaCy.")
