import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

df = pd.read_csv("ml_training_data.csv")

texts = df["text"].astype(str).tolist()
labels = df["label"].astype(str).tolist()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y = to_categorical(y)

MAX_WORDS = 10000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM model
model = Sequential()
model.add(Embedding(MAX_WORDS, 128, input_length=MAX_LEN))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.4))
model.add(Dense(128, activation="relu"))
model.add(Dense(y.shape[1], activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\n===== LSTM MODEL =====")
print("Accuracy:", test_accuracy)

model.save("LSTM_model.keras")

import joblib
joblib.dump(tokenizer, "lstm_tokenizer.pkl")
joblib.dump(label_encoder, "lstm_label_encoder.pkl")
