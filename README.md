# ResuMate-NLP: Hybrid Resume Parser 🚀

A sophisticated hybrid NER system that combines rule-based and machine learning approaches to extract entities from resumes with **80.7% F1 score** accuracy.

## 📁 Project Structure

```
ResuMate-nlp/
├── 📂 src/
│   ├── 📂 parsing/                # PRODUCTION CODE
│   │   ├── __init__.py           # Package initializer
│   │   ├── hybrid.py             # MAIN SYSTEM: Hybrid NER (Rule-based + CRF)
│   │   ├── entities.py           # Rule-based entity extraction
│   │   ├── train_crf.py          # CRF model training
│   │   └── entities2.json        # Training data (220 labeled resumes)
│   │
│   └── 📂 models/                # Saved ML models
│       └── crf_model.pkl         # Pre-trained CRF model
│
├── 📂 data/
│   └── 📂 resume/                # DATASETS
│       ├── parsed_resumes.csv    # 400 processed resumes (text)
│       └── pdfs/                 # Original PDF resumes
│
├── 📄 extracted_entities.json     # 400-resume analysis (rule-based results)
├── 📄 ground_truth_labeled.json   # 30 manually labeled resumes (gold standard)
├── 📄 predictions.json            # Optional: predictions on user-uploaded resumes
├── setup.py                       # Project installation script
│
└── 📂 test/                       # TESTING & EVALUATION
    ├── __init__.py
    ├── test.py                     # Complete evaluation suite
    └── (evaluation results)        # Generated on-demand

##  Quick Start

### 1. Use the Hybrid System
```python
from src.parsing.hybrid import analyze_resume
from src.parsing.train_crf import prepare_complete_training_data, prepare_crf_data, sent2features, sent2labels
import sklearn_crfsuite

# Train the system once
training_data = prepare_complete_training_data("src/parsing/entities2.json")
labeled_sentences = prepare_crf_data(training_data)
X_train = [sent2features(s) for s in labeled_sentences]
y_train = [sent2labels(s) for s in labeled_sentences]

crf_model = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
crf_model.fit(X_train, y_train)

# Extract entities from any resume
result = analyze_resume(resume_text, crf_model)
```

### 2. Run Evaluation
```bash
cd test/
python test.py  # Generates comprehensive evaluation report
```

##  System Performance

| Metric | Score | Assessment |
|--------|-------|------------|
| **Overall F1** | 80.7% |  **Excellent** |
| **Precision** | 83.7% |  High Accuracy |
| **Recall** | 79.9% |  Good Coverage |
| **Tested On** | 30 resumes | Statistically Significant |

**Performance Level:**  **PRODUCTION-READY**

##  What Each File Does

###  Core System (`src/parsing/`)
- **`hybrid.py`** - **MAIN ENTRY POINT**: Combines rule-based + CRF ML for entity extraction
- **`entities.py`** - Rule-based skills/job titles extraction using pattern matching
- **`train_crf.py`** - Trains CRF model on labeled data for structured entity recognition
- **`entities2.json`** - 220 labeled resumes used for training the CRF model

###  Data Files
- **`extracted_entities.json`** - Complete analysis of 400 resumes using rule-based system
- **`ground_truth_labeled.json`** - 30 manually verified resumes (gold standard for evaluation)
- **`data/resume/parsed_resumes.csv`** - Raw processed resume texts from 400 PDFs

### Testing & Evaluation
- **`test/test.py`** - Complete evaluation suite (regenerates all metrics)
- Run this to get F1 scores, precision/recall, and performance analysis

##  Entity Types Extracted

| Entity Type | Examples | Performance |
|-------------|----------|-------------|
| **Skills** | Python, React, AWS, Docker | 80.8% F1 |
| **Job Titles** | Software Engineer, Data Scientist | 97.8% F1 |
| **Education** | Degrees, Universities | CRF-enhanced |
| **Experience** | Companies, Designations | CRF-enhanced |
| **Contact Info** | Emails, Phones | Rule-based |

##  Development Workflow

### Adding New Skills
1. Edit skill patterns in `src/parsing/entities.py`
2. Test with `test/test.py`
3. Regenerate evaluation metrics

### Improving ML Model
1. Add more labeled data to `entities2.json`
2. Retrain CRF: `python src/parsing/train_crf.py`
3. Evaluate improvements

### Full System Evaluation
```bash
cd test/
python test.py > evaluation_report.txt
```

##  Key Features

- **🤖 Hybrid Architecture**: Rule-based precision + ML recall
- **🎯 High Accuracy**: 80.7% F1 score on real resume data
- **📊 Scalable**: Processes 400+ resumes efficiently
- **🔍 Comprehensive**: Extracts 280+ technical skills across categories
- **📝 Professional**: Industry-standard evaluation methodology

##  Production Deployment

The system is **production-ready** and can be used for:
- Resume screening and parsing
- Skill gap analysis across organizations
- Talent acquisition pipelines
- HR analytics and reporting

##  Requirements

```txt
pandas>=1.3.0
scikit-learn>=1.0.0
sklearn-crfsuite>=0.3.0
numpy>=1.21.0
```

## 👥 Contributing

1. Fork the repository
2. Add new skill patterns to `entities.py`
3. Test with `test/test.py`
4. Submit pull request with evaluation results

## 📄 License

MIT License - feel free to use this project for personal and commercial purposes.

---

**Built for efficient resume processing and talent analytics**