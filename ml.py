import pandas as pd
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity

# --- Step 1: Load Vectorized CSV and Vectorizer ---
df = pd.read_csv("vectorized_verdic_ai.csv")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Step 2: Extract TF-IDF Columns ---
vector_columns = [col for col in df.columns if col.startswith("tfidf_")]
tfidf_matrix = df[vector_columns].values  # numpy array of embeddings

# --- Step 3: Define Preprocessing ---
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# --- Step 4: Sample Input ---
sample_title = "Robbery in residential colony"
sample_details = "A group of men broke into a house at night and stole valuables."
sample_incidentDate = "2024-12-10"
sample_location = "Bangalore"
sample_category = "Theft"
sample_section = "Section 379 IPC"

# Combine and preprocess input
user_text = ' '.join([
    preprocess(sample_title),
    preprocess(sample_details),
    preprocess(sample_incidentDate),
    preprocess(sample_location),
    preprocess(sample_category),
    preprocess(sample_section)
])

# --- Step 5: Vectorize New Input ---
sample_vector = vectorizer.transform([user_text]).toarray()

# --- Step 6: Cosine Similarity ---
similarity_scores = cosine_similarity(sample_vector, tfidf_matrix)
top_indices = similarity_scores[0].argsort()[-5:][::-1]

# --- Step 7: Show Top Matches ---
top_cases = df.loc[top_indices, ["id", "title"]].copy() if "id" in df.columns else df.loc[top_indices, ["title"]].copy()
top_cases["similarity_score"] = similarity_scores[0][top_indices] * 100

print("\nTop 5 Similar Cases:\n")
print(top_cases)
