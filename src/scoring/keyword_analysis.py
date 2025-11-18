from keybert import KeyBERT
import pandas as pd
import os

# === 1️⃣ Load your scoring results (with similarity_score) ===
path = os.path.join(os.path.dirname(__file__), "../../data/resume/match_results.csv")
df = pd.read_csv(path)

# === 2️⃣ Input job description again ===
job_description = input("Paste the job description:\n")

# === 3️⃣ Initialize KeyBERT model ===
kw_model = KeyBERT()

# Extract top job keywords
job_keywords = [kw for kw, _ in kw_model.extract_keywords(job_description, top_n=10)]
print(f"\n🧠 Job Keywords: {job_keywords}\n")

# === 4️⃣ Compare each resume and find missing keywords ===
missing_keywords = []
coverage_scores = []

for i, row in df.iterrows():
    resume_text = str(row["cleaned_resume"])
    resume_keywords = [kw for kw, _ in kw_model.extract_keywords(resume_text, top_n=15)]
    
    # Which job keywords are missing from resume?
    missing = [kw for kw in job_keywords if kw not in resume_keywords]
    missing_keywords.append(", ".join(missing))

    # Simple coverage score (how many job keywords appear)
    match_ratio = 1 - (len(missing) / len(job_keywords))
    coverage_scores.append(match_ratio)

# === 5️⃣ Combine results ===
df["missing_keywords"] = missing_keywords
df["keyword_match_score"] = coverage_scores

# === 6️⃣ Combine with similarity for final weighted score ===
# Combine similarity and keyword match score for final score (80% similarity, 20% keyword match)
df["final_score"] = (df["similarity_score"] * 0.8) + (df["keyword_match_score"] * 0.2)

# === 7️⃣ Save output ===
out_path = os.path.join(os.path.dirname(__file__), "../../data/resume/final_results.csv")
df.to_csv(out_path, index=False)
print(f"✅ Done! Saved final results to: {out_path}")

# === 8️⃣ Show top 5 matches sorted by final score ===
print("\n🏆 Top 5 Resumes:")
print(df.sort_values(by="final_score", ascending=False)[["filename", "final_score", "missing_keywords"]].head(5))
