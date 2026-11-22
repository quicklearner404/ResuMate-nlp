# ResuMate-NLP: Resume Parser and Job Matcher

[![CI](https://github.com/quicklearner404/ResuMate-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/quicklearner404/ResuMate-nlp/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97-DistillbertNER-yellow)](https://huggingface.co/quicklearner404/DistillbertNER)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ResuMate extracts structured fields from PDF resumes (name, email, phone, job titles, companies, degrees, colleges, locations and skills) and compares the resume against a job description in a Streamlit app.

It is a **hybrid system**:
- a **DistilBERT token classifier** for free-text fields (name, designation, company, degree, college, location), fine-tuned on 220 hand-annotated resumes plus 2,000 template-generated ones
- **rules** for structured fields: regex for email and phone, and a phrase matcher with a skill dictionary.

![ResuMate app analysing the sample resume against the sample job description](docs/app_screenshot.png)

## Results

Evaluated on **30 hand-annotated test resumes** that are not in the training data, with the published model [`quicklearner404/DistillbertNER`](https://huggingface.co/quicklearner404/DistillbertNER). Everything below is reproducible with `python evaluation/evaluate.py` and re-checked in CI.

| Matching | Precision | Recall | F1 |
|---|---|---|---|
| Lenient: substring match, e.g. "London" vs "London UK" | 81.7% | 83.1% | **82.4%** |
| Exact: normalized strings must be equal | 57.8% | 59.9% | **58.8%** |

The large gap between the two rows comes from boundary errors. The model usually finds the right entity but includes or drops a word ("MBA, Stockholm School" vs "Stockholm School of Economics").

Per field, lenient matching ([full report with every error](evaluation/results/report.md)):

| Field | Method | Precision | Recall | F1 |
|---|---|---|---|---|
| Name | DistilBERT | 96.4% | 100.0% | 98.2% |
| Designation | DistilBERT | 94.1% | 84.2% | 88.9% |
| Skills | Phrase matcher | 95.6% | 82.7% | 88.7% |
| Degree | DistilBERT + rules | 79.0% | 96.8% | 87.0% |
| Location | DistilBERT | 100.0% | 75.0% | 85.7% |
| College | DistilBERT | 51.8% | 93.5% | 66.7% |
| Company | DistilBERT | 78.3% | 52.9% | 63.2% |

### Evaluation audit: what changed and why

An earlier version of this README reported **93.90% weighted F1 with 100% precision**. Those numbers came from a flawed evaluation script and have been withdrawn.

1. **False positives were never counted.** For each gold field the old script recorded either the correct label or `"MISSED"`, so a wrong prediction (for example *"washington"* returned as the company) could not appear in the precision calculation. Precision was 100% by construction. The new scorer ([`evaluation/evaluate.py`](evaluation/evaluate.py)) counts every unmatched prediction as a false positive. [`tests/test_evaluation.py`](tests/test_evaluation.py) pins this behaviour.
2. **One field match counted as the whole field.** Finding one of two companies counted as fully correct. Each gold value is now scored separately.
3. **A training-curve graph was partly synthetic.** When F1 values were missing from the training log, the plotting script filled them with `np.linspace` estimates. The graph and its script have been removed. The loss curve below is kept: its values are transcribed from the Colab training logs.
4. **The published checkpoint does not reproduce the old numbers**, even with the old scorer (90.9% versus 93.9% weighted F1), so the original figures probably came from a different checkpoint. All numbers here are for the published model, which anyone can download and check.

### A decoding bug found by the new evaluation

The error report showed entity fragments such as *"care"* for **Careem** and *"kubernet"* for **Kubernetes**. Entities were being assembled from WordPiece sub-tokens, so when the model labelled only the first piece of a word, the entity was cut off mid-word.

`BertExtractor.predict` now decodes at **word level**: each word takes the label of its first sub-token and its full character span, which matches how the model was trained. Results on the same model and test set:

| | Lenient F1 | Exact F1 | Name F1 | Location precision |
|---|---|---|---|---|
| Sub-token decoding (before) | 81.1% | 58.0% | 92.9% | 95.5% |
| Word-level decoding (now) | **82.4%** | **58.8%** | **98.2%** | **100%** |

[`tests/test_decoding.py`](tests/test_decoding.py) reproduces the bug with a stub model: it fails on the old decoder (`['Care']`) and passes on the new one (`['Careem']`).

### Known weaknesses

- **The training data is mostly synthetic.** Only 220 of the 2,216 training resumes are real. The other 2,000 come from a template generator with fixed layouts, so the model has seen little variety in real resume formatting. This is the most likely cause of the two weakest fields below, and of noisier output on real PDFs than on the test set (for example a skill name returned as a company).
- **College precision (51.8%)**: employers are often labelled as colleges (Arbisoft, Nielsen, Raytheon). This is a model error, not a decoding error.
- **Company recall (52.9%)**: about half of employers are missed, especially in short bullet-point layouts.
- **Small test set**: 30 resumes, so per-field numbers can move several points with a single resume.

The most promising next step is more real annotated resumes rather than more synthetic ones.

### Training

DistilBERT (`distilbert-base-uncased`) fine-tuned for token classification with 13 BIO labels on 2,216 resumes after cleaning: **220 hand-annotated resumes** (`data/train/annotated_resumes.json`) plus **2,000 template-generated resumes** (`data/train/synthetic_resumes.json`). Training ran for 15 epochs. Validation loss was lowest at epoch 4 and rose afterwards, a sign of overfitting:

![Training and validation loss](docs/training_curves.png)

To rebuild the training data and retrain (GPU recommended):

```bash
python training/generate_synthetic.py   # writes synthetic + combined data to data/train/
python training/clean_data.py           # -> data/train/cleaned_training_data.json
python training/train_distilbert.py     # -> models/final_model/
```

## Quick start

```bash
git clone https://github.com/quicklearner404/ResuMate-nlp.git
cd ResuMate-nlp
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app/streamlit_app.py
```

The model downloads from the Hugging Face Hub on first run. To use a local copy, put it in `models/final_model/`.

**Reproduce the evaluation and run the tests:**

```bash
python evaluation/evaluate.py      # writes evaluation/results/report.md and metrics.json
pytest
```

## Project structure

```text
resumate/                  the library used by the app and the evaluation
  extraction/
    hybrid_parser.py       DistilBERT extractor + merge with rule-based fields
    rules.py               regex (email, phone) and skill dictionary (spaCy PhraseMatcher)
    pdf_parser.py          PDF text extraction, cleaning and fallbacks
  matching/
    role_classifier.py     resume–job match score and missing-skill suggestions
app/                       Streamlit app (streamlit_app.py) and sample inputs
training/                  synthetic data generation, cleaning, fine-tuning, loss plot
evaluation/                entity-level scorer and committed results
data/
  train/                   annotated, synthetic, combined and cleaned training data
  test/                    30 hand-annotated test resumes
  resume/                  400 generated PDF resumes and job-matching outputs
scripts/                   batch jobs: embedding-based matching, keyword analysis, CSV to PDF
experiments/               earlier approaches: CRF tagger, job-title classifiers (see its README)
tests/                     scorer, decoding and end-to-end pipeline tests
docs/                      figures
```

The original layout from the course project is preserved on the [`dev` branch](https://github.com/quicklearner404/ResuMate-nlp/tree/dev).

## Team

Built as a university NLP project.

- **Fatima Kashif** (project lead): NER pipeline, DistilBERT training, synthetic data generation and cleaning, rule-based extraction, PDF parsing, the CRF experiments and the evaluation; the Streamlit app with Bilal
- **Bilal Asghar**: job-title classifiers (logistic regression, SVM, LSTM), skill recommendations, Streamlit frontend and parser integration
- **Muhammad Abdullah**: embedding-based resume–job matching and keyword analysis

## License

MIT
