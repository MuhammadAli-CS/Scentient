"""Repository-relative paths so the app and CLI work from any working directory."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

SMILES_CSV = os.path.join(DATA_DIR, "sample_smiles.csv")
FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")
CLEANED_FEATURES_CSV = os.path.join(DATA_DIR, "cleaned_features.csv")
LABELS_CSV = os.path.join(DATA_DIR, "odor_labels.csv")
FINAL_DATASET_CSV = os.path.join(DATA_DIR, "final_dataset.csv")
IMPORTANCE_CSV = os.path.join(DATA_DIR, "feature_importance.csv")
PERFUMES_CSV = os.path.join(DATA_DIR, "fragrantica_cleaned.csv")
METRICS_JSON = os.path.join(MODELS_DIR, "metrics.json")
