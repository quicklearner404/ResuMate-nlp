from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(resume_text, job_description):
    embeddings = model.encode([resume_text, job_description], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return (score + 1) * 50  # converts -1 to 1 → 0 to 100%

