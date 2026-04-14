import os
import argparse
import logging
from parse_smiles import parse_smiles
from featurize import featurize
from clean_features import clean_features
from select_top_features import select_top_features
from train_model import train_model
from predict import predict_smiles
from dupe_finder import DupeFinder

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# File paths
SMILES_CSV = "data/sample_smiles.csv"
FEATURES_CSV = "data/features.csv"
CLEANED_FEATURES_CSV = "data/cleaned_features.csv"
LABELS_CSV = "data/odor_labels.csv"
FINAL_DATASET_CSV = "data/final_dataset.csv"
IMPORTANCE_CSV = "data/feature_importance.csv"

def run_train_pipeline():
    logging.info("Starting ML Training Pipeline...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    logging.info("1. Parsing SMILES...")
    molecules = parse_smiles(SMILES_CSV)
    
    logging.info("2. Extracting Mordred Features...")
    featurize(molecules, FEATURES_CSV)
    
    logging.info("3. Cleaning Features...")
    clean_features(FEATURES_CSV, CLEANED_FEATURES_CSV)
    
    logging.info("4. Selecting Top Features...")
    select_top_features(CLEANED_FEATURES_CSV, LABELS_CSV, FINAL_DATASET_CSV, IMPORTANCE_CSV, top_n=250)
    
    logging.info("5. Training Random Forest Model...")
    train_model(FINAL_DATASET_CSV)
    
    logging.info("Pipeline Complete!")

def cmd_predict(smiles):
    logging.info(f"Predicting odor for SMILES: {smiles}")
    try:
        label = predict_smiles(smiles)
        print(f"\n======== PREDICTION ========")
        print(f"SMILES: {smiles}")
        print(f"Odor Descriptors: {label}")
        print(f"============================")
    except Exception as e:
        logging.error(f"Prediction failed: {e}")

def cmd_dupe(query, top_n=10):
    logging.info(f"Searching for dupes matching: {query}")
    try:
        finder = DupeFinder()
        dupes = finder.find_dupes(query, top_n=top_n)
        if dupes is not None:
            print(f"\n======== DUPE FINDER ========")
            print(f"Top {top_n} matches for '{query}':")
            print(dupes[['Perfume', 'Brand', 'Similarity', 'Rating Value']].to_string(index=False))
            print(f"=============================")
        else:
            print(f"Could not find a match for '{query}' in our database.")
    except Exception as e:
        logging.error(f"Dupe matching failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Scentient - AI-Powered Fragrance Analysis")
    parser.add_argument("--train", action="store_true", help="Run the end-to-end ML training pipeline.")
    parser.add_argument("--predict", type=str, help="Predict the odor for a given SMILES string.")
    parser.add_argument("--dupe", type=str, help="Find similar perfumes matching the given name.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of dupe matches to return (default 10).")
    
    args = parser.parse_args()
    
    if args.train:
        run_train_pipeline()
    elif args.predict:
        cmd_predict(args.predict)
    elif args.dupe:
        cmd_dupe(args.dupe, args.top_n)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
