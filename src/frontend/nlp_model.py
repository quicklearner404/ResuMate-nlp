import re
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

VEC_PATH = "tfidf_vectorizer.pkl"
MODEL_PATH = "logistic_model.pkl"

vectorizer = joblib.load(VEC_PATH)
ml_model = joblib.load(MODEL_PATH)

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("classes.npy", allow_pickle=True)

def predict_job_title(text):
    X_vec = vectorizer.transform([text])
    pred_label = ml_model.predict(X_vec)[0]
    return pred_label

def ml_score(resume_text, job_description):

    resume_pred = predict_job_title(resume_text)
    job_pred = predict_job_title(job_description)


    match_score = 1.0 if resume_pred == job_pred else 0.0


    resume_vec = vectorizer.transform([resume_text])
    job_vec = vectorizer.transform([job_description])

    resume_conf = np.max(ml_model.predict_proba(resume_vec))
    job_conf = np.max(ml_model.predict_proba(job_vec))

    # 4️⃣ Final score = match score + confidence average
    final_score = (match_score * 0.6) + ((resume_conf + job_conf) / 2 * 0.4)

    return round(final_score * 100, 2)


def extract_keywords(text):
    text = text.lower()
    keywords = re.findall(r"[a-zA-Z\+\#\.]{2,}", text)

    stopwords = {"the", "with", "and", "for", "you", "your", "are", "will", "have", "to", "of"}
    keywords = [k for k in keywords if k not in stopwords]

    return set(keywords)

def generate_recommendations(resume_text, job_description):
    job_keywords = extract_keywords(job_description)
    resume_keywords = extract_keywords(resume_text)

    missing = job_keywords - resume_keywords

    if not missing:
        return "Great! Your profile strongly matches the job requirements."

    return (
            "Consider improving these areas to increase your compatibility: "
            + ", ".join(missing)
    )
