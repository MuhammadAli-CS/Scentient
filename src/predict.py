import pandas as pd
import numpy as np
import joblib
import json
import logging
import os
from rdkit import Chem
from mordred import Calculator, descriptors

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def featurize_smiles_in_memory(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    calc = Calculator(descriptors, ignore_3D=True)
    try:
        desc_dict = calc(mol).asdict()
    except Exception as e:
        raise RuntimeError(f"Error calculating descriptors: {e}")
        
    df = pd.DataFrame([desc_dict])
    df.fillna(0, inplace=True)
    return df

def predict_smiles(smiles, models_dir="models"):
    # Load required artifacts
    try:
        with open(os.path.join(models_dir, 'clean_columns.json'), 'r') as f:
            clean_columns = json.load(f)
        
        with open(os.path.join(models_dir, 'top_features.json'), 'r') as f:
            top_features = json.load(f)
            
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        model = joblib.load(os.path.join(models_dir, 'odor_model.pkl'))
        le = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing model artifact. Ensure you have run the training pipeline first. {e}")

    # 1. Featurize
    raw_features = featurize_smiles_in_memory(smiles)

    # 2. Extract strictly clean_columns (fill missing with 0 just in case)
    # mordred should output exactly the same columns, but let's be safe
    for col in clean_columns:
        if col not in raw_features.columns:
            raw_features[col] = 0
    clean_df = raw_features[clean_columns]
    
    # 3. Convert all to numeric
    clean_df = clean_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 4. Scale
    scaled_array = scaler.transform(clean_df)
    scaled_df = pd.DataFrame(scaled_array, columns=clean_columns)

    # 5. Filter to top_features
    final_X = scaled_df[top_features]

    # 6. Predict
    pred_encoded = model.predict(final_X)
    pred_label = le.inverse_transform(pred_encoded)
    
    return pred_label[0]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        smiles = sys.argv[1]
        print(f"Predicted odor for {smiles}: {predict_smiles(smiles)}")
    else:
        print("Usage: python predict.py <SMILES>")
