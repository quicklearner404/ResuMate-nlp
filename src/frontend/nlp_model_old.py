import os
import re
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# --- PATH CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VEC_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "logistic_model.pkl")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.npy")

# --- LOAD MODELS ---
vectorizer = None
ml_model = None
model_classes = None
model_active = False

try:
    if os.path.exists(VEC_PATH) and os.path.exists(MODEL_PATH):
        vectorizer = joblib.load(VEC_PATH)
        ml_model = joblib.load(MODEL_PATH)
        model_active = True
        if os.path.exists(CLASSES_PATH):
            model_classes = np.load(CLASSES_PATH, allow_pickle=True)
except Exception as e:
    print(f"Error loading models: {e}")

# --- STRICT LOGIC (ORIGINAL) ---

def predict_job_title(text):
    if model_active and vectorizer:
        try:
            # Clean and Vectorize
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
            X_vec = vectorizer.transform([text])
            pred_idx = ml_model.predict(X_vec)[0]
            
            if model_classes is not None and pred_idx < len(model_classes):
                return str(model_classes[pred_idx])
            return str(pred_idx)
        except:
            pass
    return "Unknown"

def ml_score(resume_text, job_description):
    """
    STRICT SCORING: 
    Only gives a high score if the Predicted Job Titles match perfectly.
    """
    if not model_active: return 0.0

    resume_pred = predict_job_title(resume_text)
    job_pred = predict_job_title(job_description)

    # 1. Strict Match Check
    match_score = 1.0 if resume_pred == job_pred else 0.0

    # 2. Confidence Check
    resume_vec = vectorizer.transform([re.sub(r'[^a-zA-Z0-9\s]', '', resume_text.lower())])
    job_vec = vectorizer.transform([re.sub(r'[^a-zA-Z0-9\s]', '', job_description.lower())])

    resume_conf = np.max(ml_model.predict_proba(resume_vec))
    job_conf = np.max(ml_model.predict_proba(job_vec))

    # 3. Final Weighted Score
    final_score = (match_score * 0.6) + ((resume_conf + job_conf) / 2 * 0.4)

    return round(final_score * 100, 2)

def generate_recommendations(resume_text, job_description):
    # Basic keyword extraction
    def get_keywords(text):
        stopwords = {"the", "with", "and", "for", "to", "of", "in", "a", "an", "is"}
        words = re.findall(r"\w+", text.lower())
        return {w for w in words if w not in stopwords and len(w) > 2}

    job_keywords = get_keywords(job_description)
    resume_keywords = get_keywords(resume_text)
    missing = list(job_keywords - resume_keywords)

    if not missing: return "Great! Your profile matches."
    return "Consider improving: " + ", ".join(missing[:5])