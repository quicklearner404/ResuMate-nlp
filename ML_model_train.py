import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("ml_training_data.csv")

X = df["text"]
y = df["label"]

# Encode string labels → numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression
logreg = LogisticRegression(max_iter=2000,class_weight="balanced")
logreg.fit(X_train_vec, y_train)
logreg_preds = logreg.predict(X_test_vec)

print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy:", accuracy_score(y_test, logreg_preds))
print(classification_report(y_test, logreg_preds))

# SVM
svm = SVC(kernel="linear",class_weight="balanced")
svm.fit(X_train_vec, y_train)
svm_preds = svm.predict(X_test_vec)

print("\n===== SVM CLASSIFIER =====")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print(classification_report(y_test, svm_preds))

# Save models + vectorizer + label encoder
joblib.dump(logreg, "logreg_model.pkl")
joblib.dump(svm, "svm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModels saved successfully!")
