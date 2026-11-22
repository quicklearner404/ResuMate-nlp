import os
import re
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SETUP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VEC_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "logistic_model.pkl")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.npy")

# Lowest match score shown in the app (a floor, not an offset).
MIN_DISPLAY_SCORE = 15.0

# --- 2. TRY TO LOAD TEAM MODELS ---
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
except Exception:
    pass

# =========================================================
# 3. THE "SMART" SKILL DICTIONARY
# =========================================================
# This acts as a filter. We ONLY suggest words if they exist in this list.
SKILL_DB = {
    # Languages
    "python", "java", "c++", "c#", "javascript", "typescript", "html", "css", "sql", "nosql",
    "r", "swift", "kotlin", "php", "ruby", "go", "golang", "rust", "scala", "perl", "bash", "shell",
    # AI/ML
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "pandas", "numpy", "matplotlib",
    "seaborn", "opencv", "nlp", "computervision", "generative", "llm", "bert", "transformers",
    "huggingface", "openai", "spacy", "nltk", "tableau", "powerbi", "excel",
    # Web/Frameworks
    "react", "angular", "vue", "node", "express", "django", "flask", "fastapi", "spring", "springboot",
    "dotnet", "aspnet", "laravel", "rails", "jquery", "bootstrap", "tailwind",
    # Cloud/DevOps
    "aws", "azure", "gcp", "googlecloud", "docker", "kubernetes", "k8s", "jenkins", "circleci",
    "github", "gitlab", "bitbucket", "ansible", "terraform", "linux", "unix", "ubuntu",
    # Databases
    "mysql", "postgresql", "postgres", "mongodb", "redis", "oracle", "cassandra", "elasticsearch",
    "dynamodb", "sqlite", "mariadb",
    # Concepts/Soft Skills (That are actually useful)
    "agile", "scrum", "kanban", "waterfall", "sdlc", "rest", "api", "graphql", "soap",
    "microservices", "serverless", "oop", "mvc", "testing", "junit", "selenium", "jest",
    "cypress", "mocha", "security", "encryption", "blockchain", "iot", "bigdata", "spark", "hadoop"
}

# =========================================================
# 4. LOGIC
# =========================================================

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

def predict_job_title(text):
    """ Predicts Job Title using ML Model or Fallback. """
    if model_active and vectorizer and ml_model:
        try:
            clean_input = clean_text(text)
            X_vec = vectorizer.transform([clean_input])
            pred_idx = ml_model.predict(X_vec)[0]
            if model_classes is not None and pred_idx < len(model_classes):
                return str(model_classes[pred_idx])
            return str(pred_idx)
        except:
            pass
    
    # Fallback
    text = text.lower()
    if "data" in text and "scien" in text: return "Data Scientist"
    if "machine" in text or "learning" in text: return "ML Engineer"
    if "python" in text or "java" in text: return "Software Engineer"
    return "Professional"

def ml_score(resume_text, job_description):
    """ HYBRID SCORING: Returns HIGHER of Team Model vs Keyword Match """
    if not resume_text or not job_description: return 0.0

    scores = []
    
    # 1. Keyword Match (Baseline)
    try:
        temp_vec = TfidfVectorizer(stop_words='english')
        tfidf = temp_vec.fit_transform([resume_text, job_description])
        cosine_val = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        # Show at least MIN_DISPLAY_SCORE so an unrelated resume reads as a low
        # match rather than 0%. (Previously +15 was added to every score.)
        scores.append(max(round(cosine_val * 100, 2), MIN_DISPLAY_SCORE))
    except:
        scores.append(0.0)

    # 2. Team Model (Strict)
    if model_active:
        try:
            resume_pred = predict_job_title(resume_text)
            job_pred = predict_job_title(job_description)
            if resume_pred == job_pred:
                scores.append(95.0)
            else:
                clean_res = clean_text(resume_text)
                clean_job = clean_text(job_description)
                vec_res = vectorizer.transform([clean_res])
                vec_job = vectorizer.transform([clean_job])
                prob_res = ml_model.predict_proba(vec_res)
                prob_job = ml_model.predict_proba(vec_job)
                prob_sim = cosine_similarity(prob_res, prob_job)[0][0]
                scores.append(round(prob_sim * 100, 2))
        except:
            pass

    return min(max(scores), 98.0)

def skill_gap(resume_text, job_description):
    """Skills from SKILL_DB that the job asks for, split into (matched, missing)."""
    job_words = set(re.findall(r"\w+", job_description.lower()))
    resume_words = set(re.findall(r"\w+", resume_text.lower()))
    required = {w for w in job_words if w in SKILL_DB}
    return sorted(required & resume_words), sorted(required - resume_words)


def generate_recommendations(resume_text, job_description):
    """ 
    Smart Logic: Scans Job Description for KNOWN SKILLS (from SKILL_DB)
    and checks if they are missing from the resume.
    """
    resume_text = resume_text.lower()
    job_description = job_description.lower()

    # 1. Find all SKILLS mentioned in the JD
    # We regex split to handle punctuation like "python," or "node.js"
    job_words = set(re.findall(r"\w+", job_description))
    
    # 2. Filter JD words: Only keep words that are in our SKILL_DB
    required_skills = {w for w in job_words if w in SKILL_DB}

    # 3. Check which of those are in the Resume
    resume_words = set(re.findall(r"\w+", resume_text))
    
    # 4. Calculate Missing Skills
    missing_skills = list(required_skills - resume_words)
    
    if not missing_skills:
        # If no hard skills are missing, check if resume is too short
        if len(resume_words) < 50:
            return ["⚠️ Your resume looks a bit short. Add more details about your projects."]
        return ["✅ Great match! Your resume contains all the technical skills listed."]
    
    # Sort alphabetically for cleanliness
    missing_skills.sort()
    
    # 5. Format Output
    recs = []
    for skill in missing_skills[:6]: # Top 6 missing skills
        recs.append(f"⚠️ Job requires **{skill.capitalize()}**, but it's missing from your resume.")
        
    return recs