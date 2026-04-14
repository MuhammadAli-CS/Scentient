import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def select_top_features(features_csv, labels_csv, output_csv, importance_csv, top_n=250):
    # Load data
    features = pd.read_csv(features_csv)
    labels = pd.read_csv(labels_csv)

    # Merge on 'name'
    df = pd.merge(features, labels, on='name')

    # Separate features and target
    X = df.drop(columns=['name', 'odor'])
    y = df['odor']

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Compute mutual information scores
    mi_scores = mutual_info_classif(X, y_encoded, random_state=42)

    # Create feature ranking DataFrame
    feature_ranking = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values(by='mi_score', ascending=False)

    # Save feature importance for analysis
    feature_ranking.to_csv(importance_csv, index=False)
    logging.info(f"Feature importance saved to {importance_csv}")

    # Select top N features
    top_features = feature_ranking.head(top_n)['feature'].tolist()
    logging.info(f"Selected top {len(top_features)} features based on mutual information.")

    with open('models/top_features.json', 'w') as f:
        json.dump(top_features, f)

    # Filter original data
    final_df = df[['name', 'odor'] + top_features]

    # Save to CSV
    final_df.to_csv(output_csv, index=False)
    logging.info(f"✅ Final dataset saved to {output_csv}. Shape: {final_df.shape}")