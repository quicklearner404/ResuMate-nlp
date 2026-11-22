# Experiments

Earlier approaches, kept for reference. They are **not** part of the app or the
evaluated system, and their scripts still use the file paths they were written with.

| Folder | What it is | Author |
|---|---|---|
| `crf/` | First NER approach: a CRF (`sklearn-crfsuite`) trained on the 220 annotated resumes, combined with the rule-based extractor, plus its evaluation outputs. Replaced by fine-tuned DistilBERT. | Fatima Kashif |
| `baselines/` | Job-title classifiers on the synthetic resumes: logistic regression, linear SVM and an LSTM, plus their trained artifacts and the data generator they used. | Bilal Asghar |
| `embedding_similarity_check.py` | Sanity check for the sentence-embedding resume–job similarity in `scripts/embedding_match.py`. | Muhammad Abdullah |
