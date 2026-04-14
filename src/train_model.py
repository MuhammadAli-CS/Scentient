import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def train_model(final_dataset_csv, model_path="models/odor_model.pkl", encoder_path="models/label_encoder.pkl"):
    # Load data
    df = pd.read_csv(final_dataset_csv)

    # Separate X, y
    X = df.drop(columns=['name', 'odor'])
    y = df['odor']

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    # print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model & encoder
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    print(f"✅ Model saved to {model_path}, encoder saved to {encoder_path}")

if __name__ == "__main__":
    train_model("data/final_dataset.csv")
