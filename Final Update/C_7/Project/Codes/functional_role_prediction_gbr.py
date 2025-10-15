"""
functional_role_prediction_gbr.py

Classification-style pipeline using GradientBoostingClassifier to predict
High/Low carbon-degradation based on 'β_Glucosidase (µmol/g/h)'. This mirrors
the RandomForestClassifier example you provided but uses a GB classifier.

Outputs:
- Model saved to: gbr_functional_predictor.joblib
- Predictions saved to: functional_predictions_gbr.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# File paths / constants
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / 'Dataset' / 'Soil_microbe_dataset.csv'
OUTPUT_FILE = BASE_DIR / 'functional_predictions_gbr.csv'
MODEL_FILE = BASE_DIR / 'gbr_functional_predictor.joblib'
TARGET_COLUMN_NAME = 'β_Glucosidase (µmol/g/h)'
NEW_TARGET_NAME = 'Predicted_Functional_Role'


def convert_range(val):
    """Converts range strings (e.g., '10–20') to their mean float value."""
    if isinstance(val, str):
        if "–" in val:
            parts = val.split("–")
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                return np.nan
        try:
            return float(val)
        except:
            return np.nan
    return val


def prepare_data(df, target_col):
    """Prepare numeric features X and binary target y from dataframe.

    - Converts depth ranges to numeric
    - Builds y as High/Low based on median of target_col
    - Uses numeric features only (drops ID and the raw target)
    - Fills missing values with column means
    """
    # Convert depth ranges if present
    if 'Soil_Depth_cm' in df.columns:
        df['Soil_Depth_cm'] = df['Soil_Depth_cm'].apply(convert_range)

    # Build binary target from median
    median_activity = df[target_col].median()
    y = (df[target_col] > median_activity).astype(int)
    y = y.replace({1: 'High_C_Degradation', 0: 'Low_C_Degradation'})

    # Numeric features only; drop ID and target to avoid leakage
    X = df.select_dtypes(include=[np.number]).drop(columns=['ID', target_col], errors='ignore')

    # Drop entirely-empty columns and fill remaining NaNs with mean
    X = X.dropna(axis=1, how='all')
    X = X.fillna(X.mean())

    return X, y


if __name__ == '__main__':
    # Load
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f'Dataset not found at {INPUT_FILE}')

    df_raw = pd.read_csv(INPUT_FILE)

    # Prepare
    X, y = prepare_data(df_raw.copy(), TARGET_COLUMN_NAME)

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Gradient Boosting Classifier
    print('Starting Gradient Boosting Classifier training...')
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    print('Training Complete.')

    # Evaluate
    y_pred_test = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"\nModel Test Accuracy: {accuracy:.4f}")
    print('--- Test Set Classification Report ---')
    print(classification_report(y_test, y_pred_test))

    # Ensure plots directory exists
    PLOTS_DIR = BASE_DIR / 'plots'
    PLOTS_DIR.mkdir(exist_ok=True)

    # Confusion matrix and plot
    # Choose label order if available
    label_order = ['High_C_Degradation', 'Low_C_Degradation']
    labels = [lbl for lbl in label_order if lbl in y_test.values]
    if not labels:
        labels = list(np.unique(y_test))

    cm = confusion_matrix(y_test, y_pred_test, labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Gradient Boosting Classifier')
    plt.tight_layout()
    conf_mat_out = PLOTS_DIR / 'gbr_confusion_matrix.png'
    plt.savefig(conf_mat_out, dpi=150)
    plt.close()
    print(f'Confusion matrix saved to: {conf_mat_out}')

    # Predict for full dataset
    y_full_pred = clf.predict(X)

    # Save model
    joblib.dump(clf, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")

    # Build output dataframe: ID + features + predicted role
    df_output = df_raw[['ID']].copy()
    df_output = df_output.merge(X.reset_index(drop=True), left_index=True, right_index=True, how='left')
    df_output[NEW_TARGET_NAME] = y_full_pred

    # Save predictions
    df_output.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully generated predictions for {len(df_output)} samples.")
    print(f"File saved as {OUTPUT_FILE}. Use this file for your eco-metabolic map.")
