# src/scoring/job_matching.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os


embeddings_path = os.path.join(os.path.dirname(__file__), "../data/resume/resume_embeddings.pkl")

if not os.path.exists(embeddings_path):
    raise FileNotFoundError(" Embeddings file not found. Run embedding_test.py first!")

print("Loading resume embeddings...")
df = pd.read_pickle(embeddings_path)
print(f" Loaded {len(df)} resume embeddings\n")


job_description = input("Paste the job description here:\n> ")

#  job description to embeddings 
print("\nGenerating embedding for the job description...")
model = SentenceTransformer('all-MiniLM-L6-v2')
job_embedding = model.encode(job_description)

#  cosine similarity between job and each resume 
print("\nCalculating compatibility scores...\n")

# Extract resume embeddings as a matrix
resume_embeddings = np.vstack(df["embedding"].values)

# similarity scores
similarities = cosine_similarity([job_embedding], resume_embeddings)[0]

# Add to DataFrame
df["similarity_score"] = similarities

# Sorts by score and display top matches 
df_sorted = df.sort_values(by="similarity_score", ascending=False).reset_index(drop=True)

print(" Top 5 Matching Resumes:")
for i in range(5):
    print(f"{i+1}. {df_sorted.loc[i, 'filename'] if 'filename' in df_sorted.columns else 'Resume_'+str(i+1)} — Score: {df_sorted.loc[i, 'similarity_score']:.3f}")


output_path = os.path.join(os.path.dirname(__file__), "../data/resume/match_results.csv")
df_sorted.to_csv(output_path, index=False)

print(f"\n Full results saved to: {output_path}")
