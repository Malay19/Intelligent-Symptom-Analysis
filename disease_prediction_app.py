import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
from pathlib import Path
from typing import List, Dict, Any

# Utility functions from the original code
def standardize_symptom_names(symptom: str) -> str:
    if not isinstance(symptom, str):
        return str(symptom).lower()

    # Convert to lowercase
    symptom = symptom.lower()

    # Remove any special characters
    symptom = re.sub(r'[^a-z0-9\s]', '', symptom)

    # Replace multiple spaces with single space
    symptom = re.sub(r'\s+', ' ', symptom)

    # Trim whitespace
    symptom = symptom.strip()

    return symptom

def make_prediction(model_package, symptoms: List[str]) -> Dict:
    """Generate a disease prediction based on provided symptoms"""
    model = model_package['model']
    label_encoder = model_package['label_encoder']
    feature_names = model_package['selected_features']
    symptom_vocabulary = model_package['symptom_vocabulary']
    weighted_features = model_package.get('weighted_features', {})
    disease_symptom_profiles = model_package.get('disease_symptom_profiles', pd.DataFrame())
    precaution_mapping = model_package.get('precaution_mapping', {})

    standardized_symptoms = [standardize_symptom_names(s) for s in symptoms]
    X = np.zeros((1, len(feature_names)))
    used_symptoms = []
    weighted_to_base = {}

    for feature in feature_names:
        if '_weighted' in feature:
            base_feature = feature.replace('_weighted', '')
            weighted_to_base[feature] = base_feature

    for i, feature in enumerate(feature_names):
        if feature in standardized_symptoms:
            X[0, i] = 1
            used_symptoms.append(feature)
        elif feature in weighted_to_base and weighted_to_base[feature] in standardized_symptoms:
            base_symptom = weighted_to_base[feature]
            weight = weighted_features.get(base_symptom, 1)
            X[0, i] = weight

    y_pred = model.predict(X)[0]
    y_proba = model.predict_proba(X)[0]
    predicted_disease = label_encoder.inverse_transform([y_pred])[0]
    sorted_indices = np.argsort(y_proba)[::-1]
    alternatives = []

    for idx in sorted_indices[1:4]:  # Next 3 highest probabilities
        disease = label_encoder.inverse_transform([idx])[0]
        probability = y_proba[idx] * 100
        alternatives.append({
            'disease': disease,
            'probability': probability
        })

    explanation = []
    if not disease_symptom_profiles.empty:
        disease_symptoms = disease_symptom_profiles[disease_symptom_profiles['Disease'] == predicted_disease]

        if not disease_symptoms.empty:
            disease_symptoms = disease_symptoms.sort_values('Importance', ascending=False)
            for _, row in disease_symptoms.head(5).iterrows():
                symptom = row['Symptom']
                weight = row['Weight']
                freq = row['Frequency']
                present = symptom in standardized_symptoms
                if present:
                    impact = f"+{weight * freq:.2f}"
                    direction = "supports"
                else:
                    impact = "N/A"
                    direction = "typically present but missing"

                explanation.append({
                    'symptom': symptom,
                    'impact': impact,
                    'weight': weight,
                    'direction': direction
                })

    if not explanation and weighted_features:
        model_obj = model
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            model_obj = model.named_steps['model']
        if hasattr(model_obj, 'feature_importances_'):
            importances = model_obj.feature_importances_
            feature_imp = list(zip(feature_names, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            for feature, imp in feature_imp[:5]:
                if feature in weighted_to_base:
                    feature = weighted_to_base[feature]
                if feature in standardized_symptoms:
                    weight = weighted_features.get(feature, 1)
                    impact = f"+{weight * imp:.2f}"
                    direction = "supports"
                else:
                    impact = "0.00"
                    direction = "not present"

                explanation.append({
                    'symptom': feature,
                    'impact': impact,
                    'weight': weighted_features.get(feature, 1),
                    'direction': direction
                })

    precautions = precaution_mapping.get(predicted_disease.lower(), [])

    result = {
        'prediction': {
            'disease': predicted_disease,
            'confidence': y_proba[y_pred] * 100
        },
        'alternatives': alternatives,
        'explanation': explanation,
        'precautions': precautions
    }

    return result

def load_model_package():
    """Load the model package from file or create a placeholder"""
    # Define paths
    models_dir = Path("output/models")
    model_path = models_dir / "enhanced_model_package.joblib"

    # Create directories if they don't exist
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check if model exists
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning("Model file not found. Please run the training pipeline first.")
        return None

def generate_placeholder_data():
    """Generate placeholder data for demonstration when no model is available"""
    # Create a simple mapping for demo purposes
    symptom_list = [
        "fever", "cough", "fatigue", "shortness of breath", "headache",
        "sore throat", "nausea", "vomiting", "diarrhea", "rash",
        "joint pain", "muscle pain", "chest pain", "abdominal pain",
        "yellowing of eyes", "weight loss", "confusion", "chills"
    ]

    # A simplified disease-symptom mapping
    disease_map = {
        "common cold": ["fever", "cough", "sore throat", "fatigue"],
        "influenza": ["fever", "cough", "headache", "fatigue", "muscle pain", "chills"],
        "pneumonia": ["fever", "cough", "shortness of breath", "chest pain", "fatigue"],
        "tuberculosis": ["cough", "weight loss", "fatigue", "fever", "chest pain"],
        "covid-19": ["fever", "cough", "fatigue", "shortness of breath", "headache", "sore throat"],
        "hepatitis": ["fatigue", "yellowing of eyes", "abdominal pain", "nausea", "vomiting"],
        "malaria": ["fever", "chills", "headache", "nausea", "fatigue"]
    }

    return {"symptom_list": symptom_list, "disease_map": disease_map}

def make_demo_prediction(symptom_list, disease_map, selected_symptoms):
    """Generate a demonstration prediction when no model is available"""
    if not selected_symptoms:
        return None

    # Score diseases based on matched symptoms
    scores = {}
    for disease, symptoms in disease_map.items():
        matched = [s for s in selected_symptoms if s in symptoms]
        total = len(symptoms)
        score = len(matched) / total if total > 0 else 0
        scores[disease] = score

    # Sort by score
    sorted_diseases = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if not sorted_diseases:
        return None

    # Main prediction
    predicted_disease = sorted_diseases[0][0]
    confidence = sorted_diseases[0][1] * 100

    # Alternatives
    alternatives = []
    for i in range(1, min(4, len(sorted_diseases))):
        alternatives.append({
            'disease': sorted_diseases[i][0],
            'probability': sorted_diseases[i][1] * 100
        })

    # Explanation
    explanation = []
    disease_symptoms = disease_map.get(predicted_disease, [])
    for symptom in disease_symptoms:
        present = symptom in selected_symptoms
        explanation.append({
            'symptom': symptom,
            'impact': "+1.00" if present else "N/A",
            'weight': 1,
            'direction': "supports" if present else "typically present but missing"
        })

    # Precautions
    precautions = [
        "Consult a doctor for proper diagnosis",
        "Rest and stay hydrated",
        "Take prescribed medication",
        "Monitor your symptoms"
    ]

    result = {
        'prediction': {
            'disease': predicted_disease,
            'confidence': confidence
        },
        'alternatives': alternatives,
        'explanation': explanation,
        'precautions': precautions
    }

    return result

# Main Streamlit App
def main():
    st.set_page_config(
        page_title="Disease Prediction System",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 Medical Disease Prediction System")

    # Try to load the model
    model_package = load_model_package()

    # If model doesn't exist, use demo data
    is_demo_mode = model_package is None

    if is_demo_mode:
        st.warning("⚠️ Running in DEMO mode - predictions are simulated and not from a trained model.")
        placeholder_data = generate_placeholder_data()
        symptom_list = placeholder_data["symptom_list"]
        disease_map = placeholder_data["disease_map"]
    else:
        # Get symptom list from the model package
        symptom_list = model_package.get('symptom_vocabulary', [])

    # Clean up symptom names for display
    display_symptoms = [s.replace('symptom_', '').replace('_', ' ').title() for s in symptom_list]
    symptom_map = {display_symptoms[i]: symptom_list[i] for i in range(len(symptom_list))}

    # Create a dataframe for symptom reference
    symptom_df = pd.DataFrame({
        'No.': list(range(1, len(display_symptoms) + 1)),
        'Symptom': display_symptoms
    })

    st.markdown("### 📋 Select Your Symptoms")
    st.write("Enter your symptoms to get a disease prediction with explanation.")

    # Initialize session state for selected symptoms if not exists
    if 'selected_symptoms' not in st.session_state:
        st.session_state.selected_symptoms = []
    if 'selected_display_symptoms' not in st.session_state:
        st.session_state.selected_display_symptoms = []

    # Function to add a symptom
    def add_symptom(new_symptom):
        if new_symptom and new_symptom not in st.session_state.selected_display_symptoms:
            st.session_state.selected_display_symptoms.append(new_symptom)
            st.session_state.selected_symptoms.append(symptom_map.get(new_symptom, standardize_symptom_names(new_symptom)))

    # Function to remove a symptom
    def remove_symptom(symptom_to_remove):
        if symptom_to_remove in st.session_state.selected_display_symptoms:
            idx = st.session_state.selected_display_symptoms.index(symptom_to_remove)
            st.session_state.selected_display_symptoms.pop(idx)
            st.session_state.selected_symptoms.pop(idx)

    # Create a layout with columns
    col1, col2 = st.columns([2, 1])

    # Add the Symptom Reference Table in the sidebar
    st.sidebar.markdown("### 📝 Symptom Reference Table")
    st.sidebar.markdown("Use this table to identify symptoms by their number:")

    # Set a fixed height for the dataframe container to make it scrollable
    st.sidebar.markdown("""
    <style>
    .symptom-table {
        height: 300px;
        overflow-y: auto;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create the container and display the dataframe
    st.sidebar.markdown('<div class="symptom-table">', unsafe_allow_html=True)
    st.sidebar.dataframe(symptom_df, width=None, height=None, use_container_width=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.info("👆 If the dropdown shows numbers instead of symptom names, refer to this table to find corresponding symptoms.")

    # Symptom selection
    with col1:
        # Dropdown for symptom selection
        new_symptom = st.selectbox(
            "Search and select symptoms:",
            options=[""] + [s for s in display_symptoms if s not in st.session_state.selected_display_symptoms],
            key="symptom_select"
        )

        # Add button
        if st.button("Add Symptom"):
            add_symptom(new_symptom)

        # Show selected symptoms
        st.markdown("### Your Selected Symptoms:")
        if not st.session_state.selected_display_symptoms:
            st.info("No symptoms selected yet. Add some symptoms to get a prediction.")
        else:
            for i, symptom in enumerate(st.session_state.selected_display_symptoms):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"{i+1}. {symptom}")
                with col_b:
                    if st.button("Remove", key=f"remove_{i}"):
                        remove_symptom(symptom)

    # Prediction button and input for custom symptom
    with col2:
        st.markdown("### 🔍 Add Custom Symptom")
        custom_symptom = st.text_input("Enter a custom symptom not in the list:", key="custom_symptom")

        if st.button("Add Custom Symptom"):
            if custom_symptom:
                add_symptom(custom_symptom.title())

        # Add a small reference table in the UI as well
        st.markdown("### 📊 Quick Symptom Reference")
        # Create a smaller reference table showing the first few symptoms
        small_df = symptom_df.head(5)
        st.dataframe(small_df, use_container_width=True, height=150)
        st.caption("See sidebar for full symptom list")

        st.markdown("### ⚡ Get Prediction")
        predict_clicked = st.button("Predict Disease", type="primary")

    # Process prediction if clicked
    if predict_clicked and st.session_state.selected_symptoms:
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")

        with st.spinner("Analyzing symptoms..."):
            if is_demo_mode:
                result = make_demo_prediction(
                    placeholder_data["symptom_list"],
                    placeholder_data["disease_map"],
                    st.session_state.selected_symptoms
                )
            else:
                result = make_prediction(model_package, st.session_state.selected_symptoms)

        if result:
            # Display prediction
            col_res1, col_res2 = st.columns([1, 1])

            with col_res1:
                st.markdown("### 🔮 Primary Prediction")
                st.markdown(f"**Disease:** {result['prediction']['disease'].title()}")
                st.markdown(f"**Confidence:** {result['prediction']['confidence']:.2f}%")

                # Display gauge chart for confidence
                confidence = min(result['prediction']['confidence'], 100)
                st.progress(confidence/100)

                # Precautions
                st.markdown("### 💡 Recommended Precautions")
                if result['precautions']:
                    for i, precaution in enumerate(result['precautions']):
                        st.markdown(f"- {precaution}")
                else:
                    st.markdown("- Consult a healthcare professional for proper diagnosis")
                    st.markdown("- Follow medical advice")

            with col_res2:
                st.markdown("### 🔄 Alternative Possibilities")
                if result['alternatives']:
                    for alt in result['alternatives']:
                        st.markdown(f"- **{alt['disease'].title()}** ({alt['probability']:.2f}%)")
                else:
                    st.markdown("No significant alternatives found")

                # Explanation
                st.markdown("### 📝 Explanation")
                if result['explanation']:
                    for item in result['explanation']:
                        direction = item['direction']
                        impact_class = "🟢" if direction == "supports" else "🟠"
                        st.markdown(f"{impact_class} **{item['symptom'].replace('symptom_', '').title()}** ({direction})")
                else:
                    st.markdown("No detailed explanation available for this prediction")

            # Add a disclaimer
            st.markdown("---")
            st.markdown("""
            **⚠️ Medical Disclaimer:** This prediction is for informational purposes only and should not
            replace professional medical advice. Always consult with a healthcare professional for proper diagnosis and treatment.
            """)
        else:
            st.error("Unable to generate prediction. Please select different symptoms or try again.")

    elif predict_clicked:
        st.error("Please select at least one symptom before predicting.")

    # Information section at the bottom
    st.markdown("---")
    st.markdown("### ℹ️ About This System")
    st.markdown("""
    This disease prediction system uses machine learning to analyze symptoms and suggest possible diseases.

    **How to use:**
    1. Select symptoms from the dropdown or add custom symptoms
    2. Click 'Predict Disease' to see results
    3. Review the prediction, alternative possibilities, and explanations

    Remember that this tool is not a substitute for professional medical advice.
    """)

if __name__ == "__main__":
    main()