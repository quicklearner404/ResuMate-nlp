import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(resume_text, job_description):
    if not resume_text.strip() or not job_description.strip():
        return 0.0

    embeddings = model.encode([resume_text, job_description])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    return round(similarity * 100, 2)


def extract_keywords(text):
    """
    Extracts important skills & keywords from the job description.
    """
    text = text.lower()

    # very simple keyword pattern
    keywords = re.findall(r"[a-zA-Z\+\#\.]{2,}", text)

    # remove common useless words
    stopwords = {"the","with","and","for","you","your","are","will","have","to","of"}
    keywords = [k for k in keywords if k not in stopwords]

    return set(keywords)


def generate_recommendations(resume_text, job_description):
    job_keywords = extract_keywords(job_description)
    resume_keywords = extract_keywords(resume_text)

    missing_skills = job_keywords - resume_keywords

    if not missing_skills:
        return "Great! Your profile strongly matches the job requirements."

    # Convert to readable text
    missing_list = ", ".join(missing_skills)

    return f"Consider adding or improving these skills to increase your score: {missing_list}"
