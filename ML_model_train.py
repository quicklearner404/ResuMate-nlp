import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
import numpy as np


data_file = "tech_resumes.json"
resumes = []

with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        resumes.append(json.loads(line.strip()))

print(f"Loaded {len(resumes)} resumes")

texts = []
entity_features = []
labels = []

for r in resumes:
    texts.append(r["content"])


    entities = {ann['label'][0]: ann['points'][0]['text'] for ann in r['annotation']}


    feature_dict = {
        "Name": entities.get("Name", ""),
        "Company": entities.get("Companies worked at", ""),
        "Location": entities.get("Location", ""),
        "Degree": entities.get("Degree", ""),
        "College": entities.get("College Name", ""),
        "Email": entities.get("Email Address", "")
    }
    entity_features.append(feature_dict)


    labels.append(entities.get("Designation", "Unknown"))

vectorizer_text = TfidfVectorizer(max_features=5000)
X_text = vectorizer_text.fit_transform(texts)


vec_entities = DictVectorizer()
X_entities = vec_entities.fit_transform(entity_features)


from scipy.sparse import hstack

X = hstack([X_text, X_entities])


le = LabelEncoder()
y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
label_encoder = LabelEncoder()
label_encoder.fit(y_train)   # your training labels
np.save("classes.npy", label_encoder.classes_)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

