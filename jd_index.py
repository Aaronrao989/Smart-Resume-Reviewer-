import os
import json
import joblib
import numpy as np
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ART_DIR = os.path.join(os.path.dirname(__file__), "../..", "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

class JDIndex:
    def __init__(self, max_features=2000):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
        self.role_match_clf = None
        self.index = None
        self.meta = []

    def build_from_csv(self, csv_path):
        # Load CSV
        df = pd.read_csv(csv_path)
        print("Columns in CSV:", df.columns.tolist())

        # Use .get() to handle missing columns
        records = []
        for _, row in df.iterrows():
            job_position = str(row.get("job_position", "")).strip()
            if job_position == "":
                continue  # skip rows without job position

            text_parts = [
                str(row.get("job_position", "")),
                str(row.get("relevant_skills", "")),
                str(row.get("required_qualifications", "")),
                str(row.get("job_responsibilities", "")),
                str(row.get("ideal_candidate_summary", ""))
            ]
            text = " ".join([part for part in text_parts if part.strip() != ""])
            records.append({"text": text, "job_position": job_position})

        if len(records) == 0:
            raise ValueError("No valid records found in CSV.")

        # Fit TF-IDF
        X = self.vectorizer.fit_transform([rec["text"] for rec in records])
        y = [rec["job_position"] for rec in records]

        # Train classifier
        self.role_match_clf = LogisticRegression(
            max_iter=200, solver="saga", multi_class="multinomial"
        ).fit(X, y)

        # Save vectorizer & classifier
        joblib.dump(self.vectorizer, os.path.join(ART_DIR, "tfidf_vectorizer.pkl"))
        joblib.dump(self.role_match_clf, os.path.join(ART_DIR, "tfidf_job_match.pkl"))

        # Build FAISS index safely
        X_dense = X.astype(np.float32).toarray()
        d = X_dense.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(X_dense)

        # Save FAISS index & meta
        self.meta = records
        faiss.write_index(self.index, os.path.join(ART_DIR, "faiss_index.bin"))
        with open(os.path.join(ART_DIR, "faiss_meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

        print(f"Artifacts saved in {ART_DIR}, total records: {len(records)}")

    def load(self):
        self.index = faiss.read_index(os.path.join(ART_DIR, "faiss_index.bin"))
        with open(os.path.join(ART_DIR, "faiss_meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.vectorizer = joblib.load(os.path.join(ART_DIR, "tfidf_vectorizer.pkl"))
        self.role_match_clf = joblib.load(os.path.join(ART_DIR, "tfidf_job_match.pkl"))

    def query(self, text, k=3):
        vec = self.vectorizer.transform([text]).astype(np.float32).toarray()
        D, I = self.index.search(vec, k)
        results = []
        for idx, score in zip(I[0], D[0]):
            job = self.meta[idx]["job_position"] if idx < len(self.meta) else "Unknown"
            results.append({"job_position": job, "score": float(score)})
        return results

    def match_role(self, text):
        vec = self.vectorizer.transform([text])
        proba = self.role_match_clf.predict_proba(vec)[0]
        idx = proba.argmax()
        role = self.role_match_clf.classes_[idx]
        return role, float(proba[idx])

   

        
       


