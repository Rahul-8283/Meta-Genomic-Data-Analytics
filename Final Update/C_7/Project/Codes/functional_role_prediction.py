import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib # Used to save the model itself

# Define file paths
INPUT_FILE = "Dataset\Soil_microbe_dataset.csv"
OUTPUT_FILE = "functional_predictions.csv"
MODEL_FILE = "rf_functional_predictor.joblib"
TARGET_COLUMN_NAME = 'β_Glucosidase (µmol/g/h)'
NEW_TARGET_NAME = 'Predicted_Functional_Role'


# --- 1. Data Cleaning and Preparation Functions ---

def convert_range(val):
    """Converts range strings (e.g., '10–20') to their mean float value."""
    if isinstance(val, str):
        if "–" in val: # Handles the long dash character
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
    """Cleans data, creates target, and separates features (X) and target (y)."""
    
    # Apply range conversion to 'Soil_Depth_cm'
    if 'Soil_Depth_cm' in df.columns:
        df['Soil_Depth_cm'] = df['Soil_Depth_cm'].apply(convert_range)
    
    # 1A. Define Target (y): High/Low Carbon Degradation
    median_activity = df[target_col].median()
    y = (df[target_col] > median_activity).astype(int)
    y = y.replace({1: 'High_C_Degradation', 0: 'Low_C_Degradation'})
    
    # 1B. Define Features (X): Drop ID and the raw target column (to prevent data leakage)
    X = df.select_dtypes(include=[np.number]).drop(columns=['ID', target_col], errors='ignore')
    
    # 1C. Handle Missing Data
    X = X.dropna(axis=1, how='all')
    X = X.fillna(X.mean())
    
    return X, y

# --- 2. Main Execution Block ---

if __name__ == "__main__":
    
    # Load data
    df_raw = pd.read_csv(INPUT_FILE)
    
    # Prepare features (X) and target (y)
    X, y = prepare_data(df_raw.copy(), TARGET_COLUMN_NAME)
    
    # Split data for evaluation (20% test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the final model
    print("Starting Random Forest Model Training...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Training Complete.")
    
    # Evaluate model performance (on the test set)
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"\nModel Test Accuracy: {accuracy:.4f} (Expected ~0.98)")
    print("--- Test Set Classification Report ---")
    print(classification_report(y_test, y_pred_test))
    
    # Predict functional role for THE ENTIRE DATASET
    # We use the full feature set 'X' to predict 'y_full_pred'
    y_full_pred = model.predict(X)
    
    # Save the model (optional, but good practice)
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    
    # --- 3. Create and Save Final Output CSV ---
    
    # Combine original features (X), the derived prediction (y_full_pred), and the original ID column
    df_output = df_raw[['ID']].copy() # Start with ID
    df_output = df_output.merge(X.reset_index(drop=True), left_index=True, right_index=True, how='left')
    
    # Add the predicted functional role column
    df_output[NEW_TARGET_NAME] = y_full_pred
    
    # Save the final file for eco-metabolic mapping
    df_output.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSuccessfully generated predictions for {len(df_output)} samples.")
    print(f"File saved as {OUTPUT_FILE}. Use this file for your eco-metabolic map.")