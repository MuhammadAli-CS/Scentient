import argparse
import logging
from dupe_finder import DupeFinder
from pipeline import run_train_pipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def cmd_predict(smiles):
    logging.info(f"Predicting odor for SMILES: {smiles}")
    try:
        from predict import predict_smiles
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
            print(dupes[['Perfume', 'Brand', 'Similarity (%)', 'Rating Value']].to_string(index=False))
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
