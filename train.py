# train.py
# EMAIL SPAM CLASSIFIER (83K DATASET)
# Model: Logistic Regression

import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 65)
print("ADVANCED EMAIL SPAM CLASSIFIER - 83K DATASET")
print("Model: Logistic Regression")
print("=" * 65)

# 1. LOAD DATASET
print("\n[1/8] Loading dataset...")
df = pd.read_csv("spam.csv")

# Keep needed columns
df = df[['label', 'text']]
df.columns = ['label', 'message']
print("Total Rows:", len(df))
print("Labels:", df['label'].unique())

# 2. CLEAN TEXT FUNCTION
def clean_text(text):
    text = str(text).lower()
    # remove noisy dataset tokens
    text = text.replace("escapenumber", " ")
    text = text.replace("escapelong", " ")
    text = text.replace("escapeurl", " ")
    # remove urls
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # remove symbols / digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. CLEAN DATA
print("\n[2/8] Cleaning data...")
df.dropna(inplace=True)

before = len(df)
df.drop_duplicates(subset=['message'], inplace=True)
after = len(df)
print("Duplicates Removed:", before - after)

# Apply cleaning
df['clean_message'] = df['message'].apply(clean_text)
print("Text cleaned successfully.")

# 4. LABEL DISTRIBUTION
print("\n[3/8] Class distribution:")
print(df['label'].value_counts())

# 5. TF-IDF FEATURES
print("\n[4/8] Creating TF-IDF vectors...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95)
X = vectorizer.fit_transform(df['clean_message'])
y = df['label']
print("Feature Shape:", X.shape)

# 6. TRAIN / TEST SPLIT
print("\n[5/8] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y)
print("Train Size:", len(y_train))
print("Test Size :", len(y_test))

# 7. TRAIN MODEL
print("\n[6/8] Training Logistic Regression...")
model = LogisticRegression(
    max_iter=2000,
    C=1.5,
    n_jobs=-1)
model.fit(X_train, y_train)

# 8. EVALUATE
print("\n[7/8] Evaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(acc * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# 9. SAVE FILES
print("\n[8/8] Saving files...")
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "spam_vectorizer.pkl")
print("Saved: spam_model.pkl")
print("Saved: spam_vectorizer.pkl")

# 10. SAMPLE TESTING
print("\nSample Predictions:\n")
samples = [
    "We received your application for Software Engineering Intern.",
    "Congratulations! You won 50000 rupees click now!",
    "Please attend the team meeting tomorrow at 10 AM.",
    "Limited time loan offer apply immediately."]

for text in samples:
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "SPAM" if pred == 1 else "HAM"
    confidence = prob[pred] * 100
    print("Email :", text)
    print("Result:", label, f"({confidence:.2f}%)")
    print("-" * 55)
print("\nTRAINING COMPLETE ")