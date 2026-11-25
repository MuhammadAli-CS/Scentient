import json
import logging
import os

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from paths import MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def select_top_features(features_csv, labels_csv, output_csv, importance_csv, top_n=250, models_dir=MODELS_DIR):
    features = pd.read_csv(features_csv)
    labels = pd.read_csv(labels_csv)
    df = pd.merge(features, labels, on='name')

    X = df.drop(columns=['name', 'odor'])
    y = LabelEncoder().fit_transform(df['odor'])

    mi_scores = mutual_info_classif(X, y, random_state=42)
    feature_ranking = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
    feature_ranking = feature_ranking.sort_values(by='mi_score', ascending=False)
    feature_ranking.to_csv(importance_csv, index=False)
    logging.info(f"Feature importance saved to {importance_csv}")

    top_features = feature_ranking.head(top_n)['feature'].tolist()
    logging.info(f"Selected top {len(top_features)} features based on mutual information.")
    with open(os.path.join(models_dir, 'top_features.json'), 'w') as f:
        json.dump(top_features, f)

    final_df = df[['name', 'odor'] + top_features]
    final_df.to_csv(output_csv, index=False)
    logging.info(f"Final dataset saved to {output_csv}. Shape: {final_df.shape}")
