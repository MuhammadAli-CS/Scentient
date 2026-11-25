"""End-to-end training: SMILES -> Mordred descriptors -> cleaning -> MI selection -> random forest."""
import logging
import os

from paths import (CLEANED_FEATURES_CSV, DATA_DIR, FEATURES_CSV, FINAL_DATASET_CSV,
                   IMPORTANCE_CSV, LABELS_CSV, MODELS_DIR, SMILES_CSV)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

TOP_N_FEATURES = 250


def run_train_pipeline(progress=None):
    """Run every stage. `progress(step, total, message)` is called before each stage."""
    from clean_features import clean_features
    from featurize import featurize
    from parse_smiles import parse_smiles
    from select_top_features import select_top_features
    from train_model import train_model

    def report(step, msg):
        logging.info(f"{step}/5 {msg}")
        if progress:
            progress(step, 5, msg)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    report(1, "Parsing SMILES")
    molecules = parse_smiles(SMILES_CSV)
    report(2, f"Computing Mordred descriptors for {len(molecules)} molecules")
    featurize(molecules, FEATURES_CSV)
    report(3, "Dropping constant and collinear descriptors")
    clean_features(FEATURES_CSV, CLEANED_FEATURES_CSV)
    report(4, "Ranking descriptors by mutual information")
    select_top_features(CLEANED_FEATURES_CSV, LABELS_CSV, FINAL_DATASET_CSV, IMPORTANCE_CSV, top_n=TOP_N_FEATURES)
    report(5, "Cross-validating and fitting the random forest")
    return train_model(FINAL_DATASET_CSV)


if __name__ == "__main__":
    run_train_pipeline()
