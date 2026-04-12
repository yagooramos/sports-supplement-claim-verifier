#!/usr/bin/env python3
"""
Claim-type classifier: TF-IDF + Logistic Regression.

Supervised text classification component that predicts claim_type
(performance, body_composition, energy_fatigue, recovery) from raw
claim text.

This is a bounded ML extension. It does not replace the deterministic
parser or reasoning layer. It exists to demonstrate a supervised
learning task on the project's domain and to strengthen the project's
coverage of Advanced Machine Learning.

Input:   raw claim text (a short natural-language supplement claim)
Target:  claim_type (one of four canonical categories)
Features: TF-IDF unigram+bigram vectors from claim text

Usage:
    python scripts/claim_type_classifier.py train
    python scripts/claim_type_classifier.py predict --claim "creatine boosts strength"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATASET_PATH = PROJECT_ROOT / "data" / "ml" / "claim_type_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "claim_type_model.joblib"
METRICS_PATH = MODEL_DIR / "claim_type_metrics.json"


def load_dataset(path: str | Path = DATASET_PATH) -> tuple[list[str], list[str]]:
    """Load the labeled claim-type dataset."""
    df = pd.read_csv(path)
    texts = df["claim_text"].tolist()
    labels = df["claim_type"].tolist()
    return texts, labels


def build_pipeline() -> Pipeline:
    """Build the TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=500,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )),
    ])


def train(_args: argparse.Namespace) -> None:
    """Train the classifier with stratified cross-validation, then save."""
    texts, labels = load_dataset()
    pipeline = build_pipeline()

    print(f"Dataset: {len(texts)} examples")
    class_counts = pd.Series(labels).value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")

    # --- Stratified 5-fold cross-validation ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_macro", "f1_weighted"]
    cv_results = cross_validate(
        pipeline, texts, labels, cv=cv, scoring=scoring, return_train_score=False,
    )

    cv_accuracy_mean = cv_results["test_accuracy"].mean()
    cv_accuracy_std = cv_results["test_accuracy"].std()
    cv_f1_macro_mean = cv_results["test_f1_macro"].mean()
    cv_f1_macro_std = cv_results["test_f1_macro"].std()
    cv_f1_weighted_mean = cv_results["test_f1_weighted"].mean()
    cv_f1_weighted_std = cv_results["test_f1_weighted"].std()

    print("\n=== Stratified 5-Fold Cross-Validation ===")
    print(f"Accuracy:     {cv_accuracy_mean:.3f} +/- {cv_accuracy_std:.3f}")
    print(f"F1 (macro):   {cv_f1_macro_mean:.3f} +/- {cv_f1_macro_std:.3f}")
    print(f"F1 (weighted): {cv_f1_weighted_mean:.3f} +/- {cv_f1_weighted_std:.3f}")

    # --- Train final model on full dataset ---
    pipeline.fit(texts, labels)
    y_pred = pipeline.predict(texts)
    report_text = classification_report(labels, y_pred)
    report_dict = classification_report(labels, y_pred, output_dict=True)

    print("\n=== Training Set Classification Report ===")
    print(report_text)

    # --- Save model and metrics ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    metrics = {
        "cv_accuracy_mean": round(cv_accuracy_mean, 4),
        "cv_accuracy_std": round(cv_accuracy_std, 4),
        "cv_f1_macro_mean": round(cv_f1_macro_mean, 4),
        "cv_f1_macro_std": round(cv_f1_macro_std, 4),
        "cv_f1_weighted_mean": round(cv_f1_weighted_mean, 4),
        "cv_f1_weighted_std": round(cv_f1_weighted_std, 4),
        "dataset_size": len(texts),
        "classes": sorted(set(labels)),
        "training_report": report_dict,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {METRICS_PATH}")


def predict(args: argparse.Namespace) -> None:
    """Load saved model and predict claim_type for a single claim."""
    if not MODEL_PATH.exists():
        print(f"Error: model not found at {MODEL_PATH}. Run 'train' first.")
        sys.exit(1)

    pipeline = joblib.load(MODEL_PATH)
    claim = args.claim
    prediction = pipeline.predict([claim])[0]
    probabilities = pipeline.predict_proba([claim])[0]
    classes = pipeline.classes_

    print(f"Claim:     {claim}")
    print(f"Predicted: {prediction}")
    print("Probabilities:")
    for cls, prob in sorted(zip(classes, probabilities), key=lambda x: -x[1]):
        print(f"  {cls}: {prob:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Claim-type classifier (TF-IDF + Logistic Regression)",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train", help="Train and evaluate the classifier")

    predict_parser = subparsers.add_parser("predict", help="Predict claim_type for a claim")
    predict_parser.add_argument("--claim", required=True, help="Raw claim text to classify")

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
