import json
import logging
import os
from functools import lru_cache

import joblib
import pandas as pd
from rdkit import Chem

from featurize import get_calculator
from paths import MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def featurize_smiles_in_memory(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    try:
        desc_dict = get_calculator()(mol).asdict()
    except Exception as e:
        raise RuntimeError(f"Error calculating descriptors: {e}")
    return pd.DataFrame([desc_dict])


@lru_cache(maxsize=4)
def _load_artifacts(models_dir, stamp):
    # `stamp` is the model file's mtime so a retrain invalidates the cache.
    try:
        with open(os.path.join(models_dir, 'clean_columns.json')) as f:
            clean_columns = json.load(f)
        with open(os.path.join(models_dir, 'top_features.json')) as f:
            top_features = json.load(f)
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        model = joblib.load(os.path.join(models_dir, 'odor_model.pkl'))
        le = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing model artifact. Run the training pipeline first. {e}")
    return clean_columns, top_features, scaler, model, le


def load_artifacts(models_dir=MODELS_DIR):
    stamp = os.path.getmtime(os.path.join(models_dir, 'odor_model.pkl'))
    return _load_artifacts(models_dir, stamp)


def predict_smiles_proba(smiles, models_dir=MODELS_DIR):
    """Return a Series of class probabilities, highest first."""
    clean_columns, top_features, scaler, model, le = load_artifacts(models_dir)

    raw = featurize_smiles_in_memory(smiles)
    raw = raw.reindex(columns=clean_columns, fill_value=0)
    raw = raw.apply(pd.to_numeric, errors='coerce').fillna(0)

    scaled = pd.DataFrame(scaler.transform(raw), columns=clean_columns)
    proba = model.predict_proba(scaled[top_features])[0]
    labels = le.inverse_transform(model.classes_)
    return pd.Series(proba, index=labels).sort_values(ascending=False)


def predict_smiles(smiles, models_dir=MODELS_DIR):
    return predict_smiles_proba(smiles, models_dir).index[0]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(predict_smiles_proba(sys.argv[1]).head(3).to_string())
    else:
        print("Usage: python predict.py <SMILES>")
