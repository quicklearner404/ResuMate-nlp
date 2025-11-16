import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv("ml_training_data.csv")

X = df["text"]
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train_vec, y_train)

logreg_preds = logreg.predict(X_test_vec)
logreg_acc = accuracy_score(y_test, logreg_preds)

print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy:", logreg_acc)
print(classification_report(y_test, logreg_preds))


svm = SVC(kernel="linear")
svm.fit(X_train_vec, y_train)

svm_preds = svm.predict(X_test_vec)
svm_acc = accuracy_score(y_test, svm_preds)

print("\n===== SVM CLASSIFIER =====")
print("Accuracy:", svm_acc)
print(classification_report(y_test, svm_preds))


import joblib
joblib.dump(logreg, "logreg_model.pkl")
joblib.dump(svm, "svm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModels saved successfully!")
