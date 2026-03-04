from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.config import MODEL_FILE, MODEL_METADATA_FILE, TRAINING_DATASET_FILE


MODEL_REGISTRY = {
    "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    ),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
}


def evaluate_model(model, x_test, y_test) -> dict:
    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
    }


def train_all(dataset_path: Path = TRAINING_DATASET_FILE) -> dict:
    dataset = pd.read_csv(dataset_path)
    x = dataset.drop(columns=["label"])
    y = dataset["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    best = {"name": None, "model": None, "metrics": None}
    all_metrics = {}

    for name, model in MODEL_REGISTRY.items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        all_metrics[name] = metrics
        if best["metrics"] is None or metrics["roc_auc"] > best["metrics"]["roc_auc"]:
            best = {"name": name, "model": model, "metrics": metrics}

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], MODEL_FILE)

    metadata = {"best_model": best["name"], "best_metrics": best["metrics"], "all_metrics": all_metrics}
    with open(MODEL_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    print(train_all())
