# tests/embedding_test.py

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os


csv_path = os.path.join(os.path.dirname(__file__), "../data/resume/parsed_resumes.csv")


print("Loading parsed resumes CSV...")
print("Looking for file at:", csv_path)
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ CSV file not found at: {csv_path}")

df = pd.read_csv(csv_path)


print(f" Loaded {len(df)} resumes")
print(df.head(3))  # show sample rows


print("\nLoading model (this may take a few seconds)...")
model = SentenceTransformer('all-MiniLM-L6-v2')


embeddings = []

print("\nGenerating embeddings...\n")
for text in tqdm(df["cleaned_resume"], desc="Processing Resumes"):
    emb = model.encode(text)
    embeddings.append(emb)


df["embedding"] = embeddings


output_path = os.path.join(os.path.dirname(__file__), "../data/resume/resume_embeddings.pkl")
df.to_pickle(output_path)

print(f"\n Done! Saved embeddings to: {output_path}")
