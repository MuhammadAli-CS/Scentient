import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from paths import MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def clean_features(input_csv, output_csv, corr_threshold=0.95, models_dir=MODELS_DIR):
    logging.info(f"Loading features from {input_csv}...")
    df = pd.read_csv(input_csv)

    names = df['name']
    df = df.drop(columns=['name'])

    # Mordred writes failed descriptors as error strings; coerce them to NaN.
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how='all')
    df = df.fillna(0)

    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        logging.info(f"Dropping {len(constant_cols)} constant columns")
    df = df.loc[:, nunique > 1]

    if df.shape[1] > 1:
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        if to_drop:
            logging.info(f"Dropping {len(to_drop)} highly correlated features")
        df = df.drop(columns=to_drop)

    kept_columns = df.columns.tolist()
    with open(os.path.join(models_dir, 'clean_columns.json'), 'w') as f:
        json.dump(kept_columns, f)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))

    df = pd.DataFrame(scaled_features, columns=df.columns)
    df.insert(0, 'name', names)
    df.to_csv(output_csv, index=False)
    logging.info(f"Cleaned features saved to {output_csv}. Shape: {df.shape}")
