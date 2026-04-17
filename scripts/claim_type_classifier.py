#!/usr/bin/env python3
"""
Claim-type classifier: TF-IDF + Logistic Regression.

This is a bounded supervised component that predicts `claim_type`
from raw claim text. It complements the deterministic parser but does
not replace the main retrieval + reasoning pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATASET_PATH = PROJECT_ROOT / "data" / "ml" / "claim_type_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "claim_type_model.joblib"
METRICS_PATH = MODEL_DIR / "claim_type_metrics.json"
_MODEL_CACHE = None


def dataset_available(path: str | Path = DATASET_PATH) -> bool:
    return Path(path).exists()


def load_dataset(path: str | Path = DATASET_PATH) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path)
    texts = [str(text).strip() for text in df["claim_text"].tolist()]
    labels = [str(label).strip() for label in df["claim_type"].tolist()]
    return texts, labels


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=500,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    C=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def train_classifier(
    dataset_path: str | Path = DATASET_PATH,
    save_model: bool = True,
    save_metrics: bool = True,
) -> dict[str, object]:
    texts, labels = load_dataset(dataset_path)
    pipeline = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_macro", "f1_weighted"]
    cv_results = cross_validate(pipeline, texts, labels, cv=cv, scoring=scoring, return_train_score=False)

    pipeline.fit(texts, labels)
    predictions = pipeline.predict(texts)
    report_dict = classification_report(labels, predictions, output_dict=True)

    metrics = {
        "cv_accuracy_mean": round(float(cv_results["test_accuracy"].mean()), 4),
        "cv_accuracy_std": round(float(cv_results["test_accuracy"].std()), 4),
        "cv_f1_macro_mean": round(float(cv_results["test_f1_macro"].mean()), 4),
        "cv_f1_macro_std": round(float(cv_results["test_f1_macro"].std()), 4),
        "cv_f1_weighted_mean": round(float(cv_results["test_f1_weighted"].mean()), 4),
        "cv_f1_weighted_std": round(float(cv_results["test_f1_weighted"].std()), 4),
        "dataset_size": len(texts),
        "classes": sorted(set(labels)),
        "training_report": report_dict,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if save_model:
        joblib.dump(pipeline, MODEL_PATH)
    if save_metrics:
        with open(METRICS_PATH, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    return {
        "model": pipeline,
        "metrics": metrics,
    }


def load_saved_model(path: str | Path = MODEL_PATH):
    model_path = Path(path)
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def load_or_train_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    model = load_saved_model()
    if model is not None:
        _MODEL_CACHE = model
        return model
    if not dataset_available():
        return None
    _MODEL_CACHE = train_classifier(save_model=False, save_metrics=False)["model"]
    return _MODEL_CACHE


def predict_claim_type(claim_text: str, model=None) -> dict[str, object] | None:
    claim = str(claim_text or "").strip()
    if not claim:
        return None

    model = model or load_or_train_model()
    if model is None:
        return None

    prediction = str(model.predict([claim])[0])
    probabilities = model.predict_proba([claim])[0]
    probability_map = {
        str(label): round(float(prob), 4)
        for label, prob in sorted(zip(model.classes_, probabilities), key=lambda item: -item[1])
    }
    top_confidence = max(probability_map.values()) if probability_map else 0.0

    return {
        "predicted_claim_type": prediction,
        "confidence": round(top_confidence, 4),
        "probabilities": probability_map,
    }


def classifier_available() -> bool:
    return dataset_available()


def _cmd_train(_args: argparse.Namespace) -> None:
    result = train_classifier(save_model=True, save_metrics=True)
    print(json.dumps(result["metrics"], indent=2))


def _cmd_predict(args: argparse.Namespace) -> None:
    result = predict_claim_type(args.claim)
    if result is None:
        raise SystemExit("Classifier unavailable or empty claim.")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Claim-type classifier (TF-IDF + Logistic Regression)")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train", help="Train and save the classifier")

    predict_parser = subparsers.add_parser("predict", help="Predict claim_type for a single claim")
    predict_parser.add_argument("--claim", required=True, help="Raw claim text")

    args = parser.parse_args()
    if args.command == "train":
        _cmd_train(args)
    elif args.command == "predict":
        _cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
