import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import joblib
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_features(input_csv, output_csv, corr_threshold=0.95):
    # Load data
    logging.info(f"Loading features from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Separate names
    names = df['name']
    df = df.drop(columns=['name'])

    # Convert all to numeric, non-numeric → NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop columns with all NaN
    df = df.dropna(axis=1, how='all')

    # Fill remaining NaNs with 0
    df = df.fillna(0)

    # Drop constant columns
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        logging.info(f"Dropping {len(constant_cols)} constant columns")
    df = df.loc[:, nunique > 1]

    # Drop highly correlated columns
    if df.shape[1] > 1:
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        if to_drop:
            logging.info(f"Dropping {len(to_drop)} highly correlated features")
        df = df.drop(columns=to_drop)

    # Save retained columns
    kept_columns = df.columns.tolist()
    with open('models/clean_columns.json', 'w') as f:
        json.dump(kept_columns, f)

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    df = pd.DataFrame(scaled_features, columns=df.columns)

    # Add names back
    df.insert(0, 'name', names)

    # Save cleaned features
    df.to_csv(output_csv, index=False)
    logging.info(f"Cleaned features saved to {output_csv}. Shape: {df.shape}")
