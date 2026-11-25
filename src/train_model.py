import json
import logging
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

from paths import CLEANED_FEATURES_CSV, FINAL_DATASET_CSV, METRICS_JSON, MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

N_ESTIMATORS = 300


def _make_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def _mi(X, y):
    return mutual_info_classif(X, y, random_state=42)


def train_model(final_dataset_csv=FINAL_DATASET_CSV, models_dir=MODELS_DIR, metrics_path=METRICS_JSON,
                cleaned_features_csv=CLEANED_FEATURES_CSV):
    """Cross-validate, then fit on every molecule and save model, encoder and metrics.

    The dataset is small, so a single 80/20 split gives a noisy score. Stratified
    k-fold predictions give every molecule exactly one out-of-fold prediction,
    which is the honest accuracy we report in the app.

    Descriptor selection is refit inside every fold on the full cleaned feature
    set, so the reported score never sees the held-out molecule's label.
    """
    df = pd.read_csv(final_dataset_csv)
    X = df.drop(columns=['name', 'odor'])
    le = LabelEncoder()
    y = le.fit_transform(df['odor'])

    X_cv, cv_model = X, _make_model()
    if cleaned_features_csv and os.path.exists(cleaned_features_csv):
        cleaned = pd.read_csv(cleaned_features_csv).set_index('name').loc[df['name']]
        X_cv = cleaned.reset_index(drop=True)
        k = min(X.shape[1], X_cv.shape[1])
        cv_model = make_pipeline(SelectKBest(_mi, k=k), _make_model())

    min_class = int(np.bincount(y).min())
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_oof = cross_val_predict(cv_model, X_cv, y, cv=cv)
    cv_acc = accuracy_score(y, y_oof)
    report = classification_report(
        y, y_oof, labels=range(len(le.classes_)), target_names=le.classes_,
        zero_division=0, output_dict=True,
    )
    logging.info(f"{n_splits}-fold CV accuracy: {cv_acc:.3f} (chance ~{1 / len(le.classes_):.3f})")

    model = _make_model()
    model.fit(X, y)
    joblib.dump(model, os.path.join(models_dir, 'odor_model.pkl'))
    joblib.dump(le, os.path.join(models_dir, 'label_encoder.pkl'))

    metrics = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "classifier": "RandomForestClassifier",
        "n_estimators": N_ESTIMATORS,
        "n_molecules": int(len(df)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(le.classes_)),
        "cv_folds": n_splits,
        "cv_accuracy": round(float(cv_acc), 4),
        "chance_accuracy": round(1 / len(le.classes_), 4),
        "per_class": {
            c: {"f1": round(report[c]["f1-score"], 3), "support": int(report[c]["support"])}
            for c in le.classes_
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Model, encoder and metrics saved to {models_dir}")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_model(), indent=2))
