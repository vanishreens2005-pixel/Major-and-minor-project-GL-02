# ============================================================
# PROJECT 2 — Fake News Detection System
# VTU 2022 Scheme | Data Science Lab | Dept. of ECE
# ============================================================

# STEP 1 — Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

# STEP 2 — Create Sample Dataset
np.random.seed(42)

real_headlines = [
    "Government announces new budget plan for 2024",
    "Scientists discover new treatment for cancer",
    "Stock market reaches all time high today",
    "New education policy introduced by ministry",
    "Climate summit held in Paris with world leaders",
    "Hospital launches free health camp for citizens",
    "Railway ministry announces new train routes",
    "Tech company reports record quarterly earnings",
    "Farmers protest ends after government talks",
    "Election commission announces voting dates",
]

fake_headlines = [
    "Aliens have landed in USA government confirms",
    "Drinking bleach cures all diseases doctors say",
    "Secret society controls entire world economy",
    "Moon is artificial object built by ancient aliens",
    "5G towers spread virus confirmed by scientists",
    "Celebrity found dead in hotel room exclusive",
    "Government puts microchips in vaccines revealed",
    "Free money scheme approved by prime minister",
    "Chocolate causes weight loss new study finds",
    "Robot uprising begins in Japan breaking news",
]

real_data = pd.DataFrame({'text': real_headlines * 50, 'label': 1})
fake_data = pd.DataFrame({'text': fake_headlines * 50, 'label': 0})
df = pd.concat([real_data, fake_data]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset Shape:", df.shape)
print(df['label'].value_counts())

# STEP 3 — Text Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

df['clean_text'] = df['text'].apply(clean_text)

# STEP 4 — TF-IDF Feature Extraction
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# STEP 5 — Train and Compare Models
models = {
    'Naive Bayes':         MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'SVM':                 LinearSVC(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:25s} → Accuracy: {acc*100:.2f}%")

# STEP 6 — Accuracy Chart
plt.figure(figsize=(10, 5))
bars = plt.bar(results.keys(), [v * 100 for v in results.values()],
               color=['#e15759','#4e79a7','#f28e2b','#76b7b2','#59a14f'])
plt.ylim(60, 105)
plt.title('Model Accuracy Comparison — Fake News Detection', fontsize=13)
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=15)
for bar, val in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val*100:.1f}%', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# STEP 7 — Confusion Matrix
best_model = LogisticRegression(max_iter=1000)
best_model.fit(X_train_tfidf, y_train)
y_pred_best = best_model.predict(X_test_tfidf)

print("\n--- Logistic Regression Classification Report ---")
print(classification_report(y_test, y_pred_best, target_names=['Fake', 'Real']))

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
disp.plot(cmap='Reds')
plt.title('Confusion Matrix — Fake News Detection')
plt.tight_layout()
plt.show()

# STEP 8 — Live Prediction
def predict_news(text):
    cleaned    = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    result     = best_model.predict(vectorized)[0]
    return "REAL NEWS ✅" if result == 1 else "FAKE NEWS ❌"

print("\n--- Live Prediction Test ---")
print(predict_news("Government announces new health policy for citizens"))
print(predict_news("Aliens confirm landing on earth breaking news"))