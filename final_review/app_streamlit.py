import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os # Import os for better path handling

# --- Configuration and Model Loading ---
MODEL_FILE = 'rf_functional_predictor.joblib'

try:
    # Check if the model file exists
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found.")
        
    # Load the trained Random Forest model
    model = joblib.load(MODEL_FILE)
    
    # Columns required for prediction (must match the training data's features)
    FEATURE_COLUMNS = [
        'Soil_pH', 'Organic_C (%)', 'Total_N (%)', 'C_N_Ratio', 
        'Soil_Depth_cm', 'Bacteria_Abundance (%)', 'Fungi_Abundance (%)', 
        'Urease (µmol/g/h)', 'CO2_Emission (µg/g/day)', 'NH4_Nitrate (µg/g)'
    ]

except FileNotFoundError:
    st.error("Error: The model file 'rf_functional_predictor.joblib' was not found. Please ensure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- Hardcoded Slider Ranges (Fixes the KeyError) ---
SLIDER_RANGES = {
    'Soil_pH': (5.5, 7.5, 6.7),
    'Organic_C (%)': (1.5, 5.0, 3.0),
    'Total_N (%)': (0.05, 0.25, 0.15),
    'C_N_Ratio': (15.0, 35.0, 25.0), 
    'Soil_Depth_cm': (0.0, 30.0, 15.0), 
    'Bacteria_Abundance (%)': (30.0, 65.0, 45.0),
    'Fungi_Abundance (%)': (15.0, 45.0, 30.0), 
    'Urease (µmol/g/h)': (5.0, 15.0, 10.0),
    'CO2_Emission (µg/g/day)': (45.0, 60.0, 52.0),
    'NH4_Nitrate (µg/g)': (80.0, 170.0, 120.0),
}
user_input_dict = {}


# --- Streamlit App Layout ---

st.set_page_config(
    page_title="Soil Functional Predictor",
    page_icon="",
    layout="wide"
)

st.title("AI-Driven Soil Functional Potential Predictor")
st.markdown("---")

# ----------------------------------------------------
# LEFT COLUMN: INPUT SIDEBAR
# ----------------------------------------------------
st.sidebar.header("Input Soil Sample Parameters")

# Create inputs for each feature in the sidebar
for feature, (min_val, max_val, default_val) in SLIDER_RANGES.items():
    user_input_dict[feature] = st.sidebar.number_input(
        f"{feature}",
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=0.01,
        help=f"Typical range: {min_val:.2f} to {max_val:.2f}"
    )


# ----------------------------------------------------
# MAIN COLUMN: PREDICTION BUTTON AND RESULTS
# ----------------------------------------------------

# Center the prediction button
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    # Button on the main screen (user requested change)
    if st.button("PREDICT FUNCTIONAL ROLE", type="primary", use_container_width=True):
        
        # --- 1. Prepare Data for Prediction ---
        input_df = pd.DataFrame([user_input_dict])
        
        # Ensure correct column order, which is CRITICAL for the model
        input_df = input_df[FEATURE_COLUMNS] 

        # --- 2. Make Prediction & Display Results ---
        try:
            prediction_label = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)
            
            # Get the probability for the predicted class
            # Note: model.classes_ might be ['Low_C_Degradation', 'High_C_Degradation']
            proba = prediction_proba[0][model.classes_ == prediction_label][0]
            
            # --- 3. Display Results ---
            st.header("Model Prediction")
            
            if prediction_label == 'High_C_Degradation':
                st.success(f"**Predicted Functional Role:** High Carbon Degradation Potential")
                st.markdown(f"**Confidence:** {proba * 100:.2f}%")
                st.markdown(
                    """
                    <div style='padding: 10px; border-radius: 5px;  border-left: 5px solid #00cc00;'>
                    High prediction suggests strong microbial (likely fungal) capacity for breaking down carbon.
                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                st.info(f"**Predicted Functional Role:** Low Carbon Degradation Potential")
                st.markdown(f"**Confidence:** {proba * 100:.2f}%")
                st.markdown(
                    """
                    <div style='padding: 10px; border-radius: 5px; background-color: #e6f7ff; border-left: 5px solid #3399ff;'>
                    Low prediction suggests limited microbial capacity for carbon breakdown relative to the dataset average.
                    </div>
                    """, unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"Prediction failed. Error: {e}")
            st.markdown("Check if your input values are reasonable.")


st.markdown("---")
st.markdown("### Input Summary")
st.dataframe(pd.DataFrame([user_input_dict]).T.rename(columns={0: "Input Value"}).style.format("{:.3f}"))