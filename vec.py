import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# --- Step 1: Load CSV ---
df = pd.read_csv(
    r"C:\Users\supre\Downloads\verdic ai.csv",
    encoding="ISO-8859-1",
    engine="python",
    quotechar='"',
    on_bad_lines='skip'
)
df.columns = df.columns.str.strip()

# --- Step 2: Ensure Required Columns ---
required_cols = ["title", "details", "incidentDate", "location", "category", "section"]
for col in required_cols:
    if col not in df.columns:
        df[col] = ''  # Fill missing columns with empty string

# --- Step 3: Preprocessing Function ---
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

for col in required_cols:
    df[col] = df[col].fillna('').apply(preprocess)

# --- Step 4: Combine Text Fields ---
df["combined_text"] = (
    (df["title"] + " ") * 2 +
    (df["details"] + " ") * 2 +
    df["incidentDate"] + " " +
    df["location"] + " " +
    df["category"] + " " +
    df["section"]
)

# --- Step 5: Vectorize with TF-IDF ---
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.85,
    min_df=2
)
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

# --- Step 6: Convert to DataFrame ---
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
)

# --- Step 7: Concatenate with Original Data ---
final_df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

# --- Step 8: Save Embeddings and Vectorizer ---
final_df.to_csv("vectorized_verdic_ai.csv", index=False)
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Vectorized data and vectorizer saved successfully.")
