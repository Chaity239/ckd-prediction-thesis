
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb

from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="CKD Prediction System",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent


# -----------------------------
# Load saved files and models
# -----------------------------
@st.cache_resource
def load_models():
    with open(BASE_DIR / "feature_columns.json", "r") as file:
        feature_columns = json.load(file)

    with open(BASE_DIR / "feature_defaults.json", "r") as file:
        feature_defaults = json.load(file)

    cat_model = CatBoostClassifier()
    cat_model.load_model(str(BASE_DIR / "catboost_ckd.cbm"))

    lgb_model = lgb.Booster(
        model_file=str(BASE_DIR / "lightgbm_ckd.txt")
    )

    tabnet_model = TabNetClassifier()
    tabnet_model.load_model(
        str(BASE_DIR / "tabnet_ckd_model.zip.zip")
    )

    return (
        feature_columns,
        feature_defaults,
        cat_model,
        lgb_model,
        tabnet_model
    )


st.title("Chronic Kidney Disease Prediction")
st.write(
    "Enter the dataset features to compare predictions from "
    "CatBoost, LightGBM, and TabNet."
)

st.warning(
    "Research prototype only — not a medical diagnosis. "
    "The displayed default values are dataset medians, not "
    "actual patient measurements. Replace them with valid "
    "patient data before making a prediction."
)

try:
    (
        feature_columns,
        feature_defaults,
        cat_model,
        lgb_model,
        tabnet_model
    ) = load_models()

except Exception as error:
    st.error(
        "The models or supporting files could not be loaded. "
        "Check that all filenames are correct and that the "
        "requirements have been installed."
    )
    st.code(str(error))
    st.stop()


# -----------------------------
# Input form
# -----------------------------
st.subheader("Patient Feature Inputs")

st.caption(
    "Enter all 51 features in the same format and units used "
    "by your original training dataset. Coded fields must use "
    "the dataset's original numeric codes."
)

with st.form("ckd_prediction_form"):
    inputs = {}

    # Divide the fields into manageable sections.
    sections = {
        "Demographic and Lifestyle Features": [
            "Age", "Gender", "Ethnicity", "SocioeconomicStatus",
            "EducationLevel", "BMI", "Smoking",
            "AlcoholConsumption", "PhysicalActivity",
            "DietQuality", "SleepQuality"
        ],
        "Medical and Family History": [
            "FamilyHistoryKidneyDisease",
            "FamilyHistoryHypertension",
            "FamilyHistoryDiabetes",
            "PreviousAcuteKidneyInjury",
            "UrinaryTractInfections"
        ],
        "Blood and Kidney Measurements": [
            "SystolicBP", "DiastolicBP", "FastingBloodSugar",
            "HbA1c", "SerumCreatinine", "BUNLevels", "GFR",
            "ProteinInUrine", "ACR",
            "SerumElectrolytesSodium",
            "SerumElectrolytesPotassium",
            "SerumElectrolytesCalcium",
            "SerumElectrolytesPhosphorus",
            "HemoglobinLevels", "CholesterolTotal",
            "CholesterolLDL", "CholesterolHDL",
            "CholesterolTriglycerides"
        ],
        "Medication and Symptoms": [
            "ACEInhibitors", "Diuretics", "NSAIDsUse",
            "Statins", "AntidiabeticMedications", "Edema",
            "FatigueLevels", "NauseaVomiting", "MuscleCramps",
            "Itching"
        ],
        "Other Health and Environmental Features": [
            "QualityOfLifeScore", "HeavyMetalsExposure",
            "OccupationalExposureChemicals", "WaterQuality",
            "MedicalCheckupsFrequency", "MedicationAdherence",
            "HealthLiteracy"
        ]
    }

    for section_name, section_features in sections.items():
        with st.expander(section_name, expanded=True):
            columns = st.columns(2)

            for index, feature in enumerate(section_features):
                if feature not in feature_columns:
                    continue

                default_value = float(
                    feature_defaults.get(feature, 0.0)
                )

                with columns[index % 2]:
                    inputs[feature] = st.number_input(
                        label=feature,
                        value=default_value,
                        format="%.4f",
                        key=f"input_{feature}"
                    )

    submitted = st.form_submit_button(
        "Predict CKD",
        type="primary",
        use_container_width=True
    )


# -----------------------------
# Make predictions
# -----------------------------
if submitted:
    missing_features = [
        feature for feature in feature_columns
        if feature not in inputs
    ]

    if missing_features:
        st.error(
            "Some features are missing from the form: "
            + ", ".join(missing_features)
        )
        st.stop()

    # Preserve the exact training feature order.
    input_df = pd.DataFrame(
        [[inputs[feature] for feature in feature_columns]],
        columns=feature_columns
    )

    try:
        # CatBoost probability of class 1 (CKD).
        cat_probability = float(
            cat_model.predict_proba(input_df)[0][1]
        )

        # LightGBM binary Booster probability.
        lgb_output = np.asarray(
            lgb_model.predict(input_df)
        ).reshape(-1)

        lgb_probability = float(lgb_output[0])

        # TabNet expects a NumPy array.
        tabnet_input = input_df.to_numpy(
            dtype=np.float32
        )

        tabnet_probabilities = tabnet_model.predict_proba(
            tabnet_input
        )

        tabnet_probability = float(
            tabnet_probabilities[0][1]
        )

        predictions = {
            "CatBoost": cat_probability,
            "LightGBM": lgb_probability,
            "TabNet": tabnet_probability
        }

        st.divider()
        st.subheader("Prediction Results")

        result_columns = st.columns(3)

        for column, (model_name, probability) in zip(
            result_columns, predictions.items()
        ):
            with column:
                st.metric(
                    label=f"{model_name} — CKD probability",
                    value=f"{probability * 100:.2f}%"
                )

                if probability >= 0.5:
                    st.error("Predicted class: CKD (1)")
                else:
                    st.success("Predicted class: Non-CKD (0)")

                st.progress(
                    max(0.0, min(1.0, probability))
                )

        st.caption(
            "The class labels above use a 0.50 probability "
            "threshold for comparison. This threshold is not "
            "a clinically validated decision threshold."
        )

        results_df = pd.DataFrame({
            "Model": list(predictions.keys()),
            "CKD Probability (%)": [
                round(probability * 100, 2)
                for probability in predictions.values()
            ],
            "Predicted Class": [
                "CKD" if probability >= 0.5 else "Non-CKD"
                for probability in predictions.values()
            ]
        })

        st.subheader("Model Comparison")
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "This is a research demonstration. Predictions "
            "must not be used alone to diagnose or treat CKD."
        )

    except Exception as error:
        st.error("Prediction failed. Please check the error below.")
        st.code(str(error))