import os
import glob
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def load_resumes(resume_dir: str):
    paths = sorted(glob.glob(os.path.join(resume_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt resumes found in: {resume_dir}")

    resumes = []
    for p in paths:
        resumes.append(
            {
                "candidate_file": os.path.basename(p),
                "text": read_text_file(p),
            }
        )
    return pd.DataFrame(resumes)


def build_training_data(job_text: str, resumes_df: pd.DataFrame):
    """
    For a simple university project: we create labels using a rule-based baseline.
    Then ML learns a model that imitates / generalizes that rule.
    Later, you can replace this with real labeled data if required.
    """

    # quick keyword baseline for labeling (weak supervision)
    must_have = ["python", "sql", "excel"]
    nice = ["pandas", "power bi", "tableau", "statistics", "visualization", "scikit", "machine learning"]

    labels = []
    for t in resumes_df["text"].str.lower().tolist():
        score = 0
        for k in must_have:
            if k in t:
                score += 2
        for k in nice:
            if k in t:
                score += 1

        # label: 1 = relevant, 0 = not relevant
        labels.append(1 if score >= 3 else 0)

    train_df = resumes_df.copy()
    train_df["label"] = labels

    # add job description as extra context feature (optional)
    # we concatenate job text to each resume to make "match" learning easier
    train_df["combined"] = (job_text + "\n\n" + train_df["text"]).astype(str)

    return train_df


def train_model(train_df: pd.DataFrame):
    X = train_df["combined"].values
    y = train_df["label"].values

    # small datasets can break stratify if only one class exists
    # we handle that safely
    stratify = y if len(set(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, (y_test, y_pred)


def score_and_rank(model, job_text: str, resumes_df: pd.DataFrame):
    combined = (job_text + "\n\n" + resumes_df["text"]).astype(str)
    # probability of class 1 (relevant)
    proba = model.predict_proba(combined)[:, 1]

    out = resumes_df.copy()
    out["relevance_score"] = proba
    out = out.sort_values("relevance_score", ascending=False).reset_index(drop=True)
    return out


def main():
    job_path = os.path.join("data", "job_description.txt")
    resumes_dir = os.path.join("data", "resumes")
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    job_text = read_text_file(job_path)
    resumes_df = load_resumes(resumes_dir)

    train_df = build_training_data(job_text, resumes_df)
    model, (y_test, y_pred) = train_model(train_df)

    # Save evaluation
    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred)

    with open(os.path.join(outputs_dir, "evaluation.txt"), "w", encoding="utf-8") as f:
        f.write("=== Classification Report ===\n")
        f.write(report + "\n\n")
        f.write("=== Confusion Matrix ===\n")
        f.write(str(cm) + "\n")

    ranked = score_and_rank(model, job_text, resumes_df)
    ranked.to_csv(os.path.join(outputs_dir, "ranked_candidates.csv"), index=False)

    print("\n✅ Done!")
    print("Saved:")
    print("- outputs/evaluation.txt")
    print("- outputs/ranked_candidates.csv")
    print("\nTop candidates:")
    print(ranked[["candidate_file", "relevance_score"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
