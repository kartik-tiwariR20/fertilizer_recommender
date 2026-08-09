import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
from tensorflow.keras.models import model_from_json

# ---------------------------------------------------------
# IMPORTANT: st.set_page_config() MUST be the very first
# Streamlit command that runs in the whole script.
# ---------------------------------------------------------
st.set_page_config(page_title="Smart Crop Advisory", page_icon="🌾", layout="centered")


# ---------------------------------------------------------
# IMPORTANT: These class definitions must match EXACTLY the
# ones used in the notebooks when the .pkl files were created.
# joblib/pickle needs these classes available to unpickle.
# ---------------------------------------------------------
class CropPipeline:
    def __init__(self, model, scaler, label_encoder):
        self.model_config = model.to_json()
        self.model_weights = model.get_weights()
        self.scaler = scaler
        self.label_encoder = label_encoder

    def predict(self, X):
        model = model_from_json(self.model_config)
        model.set_weights(self.model_weights)
        X_scaled = self.scaler.transform(X)
        probs = model.predict(X_scaled)
        idx = probs.argmax(axis=1)
        return self.label_encoder.inverse_transform(idx)

    def predict_proba(self, X):
        model = model_from_json(self.model_config)
        model.set_weights(self.model_weights)
        X_scaled = self.scaler.transform(X)
        probs = model.predict(X_scaled)
        return probs, self.label_encoder.classes_


class FertilizerPipeline:
    def __init__(self, model, preprocessor, label_encoder):
        self.model_config = model.to_json()
        self.model_weights = model.get_weights()
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder

    def predict(self, X):
        model = model_from_json(self.model_config)
        model.set_weights(self.model_weights)
        X_processed = self.preprocessor.transform(X)
        probs = model.predict(X_processed)
        idx = probs.argmax(axis=1)
        return self.label_encoder.inverse_transform(idx)


# ---------------------------------------------------------
# Load pipelines + price data (cached so they load only once)
# ---------------------------------------------------------
@st.cache_resource
def load_crop_pipeline():
    return joblib.load("crop_pipeline.pkl")


@st.cache_resource
def load_fertilizer_pipeline():
    return joblib.load("fertilizer_pipeline.pkl")


@st.cache_data(ttl=3600)
def load_prices():
    with open("fertilizer_prices.json") as f:
        return json.load(f)


crop_pipeline = load_crop_pipeline()
fertilizer_pipeline = load_fertilizer_pipeline()
price_data = load_prices()

CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# NOTE: These lists MUST exactly match the categories the fertilizer
# model's OneHotEncoder was fit on (see fertilizer_pipeline.pkl's
# preprocessor.named_transformers_['cat'].categories_). Any value not
# in this list gets silently zeroed out by handle_unknown='ignore',
# which was previously feeding the model garbage categorical data.
SOIL_TYPES = ["Clay", "Loamy", "Sandy", "Silt"]
CROP_TYPES = ["Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Tomato", "Wheat"]
GROWTH_STAGES = ["Flowering", "Harvest", "Sowing", "Vegetative"]
SEASONS = ["Kharif", "Rabi", "Zaid"]
IRRIGATION_TYPES = ["Canal", "Drip", "Rainfed", "Sprinkler"]

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🌾 Smart Crop Advisory System")
st.write("Get AI-powered crop and fertilizer recommendations based on your soil and weather conditions.")

tab1, tab2 = st.tabs(["🌱 Crop Recommendation", "🧪 Fertilizer Recommendation"])

# ===========================================================
# TAB 1 — CROP RECOMMENDATION
# ===========================================================
with tab1:
    st.subheader("Enter Soil Nutrients & Weather Conditions")

    with st.form("crop_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=50.0)
        with col2:
            P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=145.0, value=50.0)
        with col3:
            K = st.number_input("Potassium (K)", min_value=0.0, max_value=205.0, value=50.0)

        ph = st.slider("Soil pH", min_value=3.5, max_value=10.0, value=6.5, step=0.1)

        col4, col5 = st.columns(2)
        with col4:
            temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)
        with col5:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)

        crop_submitted = st.form_submit_button("Get Crop Recommendation 🌱")

    if crop_submitted:
        input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=CROP_FEATURES)
        try:
            prediction = crop_pipeline.predict(input_df)[0]
            probs, class_names = crop_pipeline.predict_proba(input_df)

            st.success(f"### Recommended Crop: **{prediction.capitalize()}** 🌾")

            top3_idx = np.argsort(probs[0])[::-1][:3]
            st.write("**Top 3 predictions:**")
            for i in top3_idx:
                st.write(f"- {class_names[i].capitalize()}: {probs[0][i] * 100:.2f}%")

            st.session_state["predicted_crop"] = prediction.capitalize()

        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")

# ===========================================================
# TAB 2 — FERTILIZER RECOMMENDATION
# ===========================================================
with tab2:
    st.subheader("Enter Soil, Crop & Environmental Details")

    with st.form("fertilizer_form"):
        col1, col2 = st.columns(2)
        with col1:
            soil_type = st.selectbox("Soil Type", SOIL_TYPES)
            crop_type = st.selectbox("Crop Type", CROP_TYPES)
            growth_stage = st.selectbox("Crop Growth Stage", GROWTH_STAGES)
        with col2:
            season = st.selectbox("Season", SEASONS)
            irrigation_type = st.selectbox("Irrigation Type", IRRIGATION_TYPES)

        st.markdown("**Soil Nutrients**")
        col3, col4, col5 = st.columns(3)
        with col3:
            nitrogen = st.number_input("Nitrogen Level", min_value=0.0, max_value=210.0, value=89.0)
        with col4:
            phosphorus = st.number_input("Phosphorus Level", min_value=0.0, max_value=120.0, value=49.0)
        with col5:
            potassium = st.number_input("Potassium Level", min_value=0.0, max_value=160.0, value=64.0)

        col6, col7 = st.columns(2)
        with col6:
            soil_ph = st.slider("Soil pH ", min_value=3.5, max_value=10.0, value=6.5, step=0.1)
            soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=80.0, value=35.0)
        with col7:
            organic_carbon = st.number_input("Organic Carbon (%)", min_value=0.0, max_value=2.0, value=0.85, step=0.05)

        st.markdown("**Weather Conditions**")
        col8, col9, col10 = st.columns(3)
        with col8:
            fert_temperature = st.number_input("Temperature (°C) ", min_value=0.0, max_value=50.0, value=25.0)
        with col9:
            fert_humidity = st.number_input("Humidity (%) ", min_value=0.0, max_value=100.0, value=60.0)
        with col10:
            # Training data used annual rainfall (mean ~1580mm, std ~810mm),
            # not a single storm event. The old 0-300 cap made every possible
            # input an extreme outlier the model had never seen.
            fert_rainfall = st.number_input("Annual Rainfall (mm) ", min_value=0.0, max_value=4000.0, value=1580.0)

        fert_submitted = st.form_submit_button("Get Fertilizer Recommendation 🧪")

    if fert_submitted:
        sample = pd.DataFrame([{
            "Soil_Type": soil_type,
            "Soil_pH": soil_ph,
            "Soil_Moisture": soil_moisture,
            "Organic_Carbon": organic_carbon,
            "Nitrogen_Level": nitrogen,
            "Phosphorus_Level": phosphorus,
            "Potassium_Level": potassium,
            "Temperature": fert_temperature,
            "Humidity": fert_humidity,
            "Rainfall": fert_rainfall,
            "Crop_Type": crop_type,
            "Crop_Growth_Stage": growth_stage,
            "Season": season,
            "Irrigation_Type": irrigation_type,
        }])

        try:
            fert_prediction = fertilizer_pipeline.predict(sample)[0]

            # --- OOD check: flag inputs far outside the training distribution ---
            num_cols = ["Soil_pH", "Soil_Moisture", "Organic_Carbon", "Nitrogen_Level",
                        "Phosphorus_Level", "Potassium_Level", "Temperature", "Humidity", "Rainfall"]
            scaler = fertilizer_pipeline.preprocessor.named_transformers_["num"]
            z_scores = (sample[num_cols].values[0] - scaler.mean_) / scaler.scale_
            extreme = [(col, z) for col, z in zip(num_cols, z_scores) if abs(z) > 2.5]

            st.success(f"### Recommended Fertilizer: **{fert_prediction}** 🧪")

            if extreme:
                extreme_list = ", ".join(f"{col.replace('_', ' ')}" for col, _ in extreme)
                st.warning(
                    f"⚠️ {extreme_list} are unusually far from typical values in the training "
                    "data. The model's prediction for this combination may be unreliable — "
                    "treat it as a rough guess rather than a confident recommendation."
                )

            # ---------------------------------------------------------
            # Show top brands + prices for the recommended fertilizer
            # ---------------------------------------------------------
            st.subheader(f"💰 Top Brands for {fert_prediction}")
            st.caption(f"Prices last updated: {price_data['last_updated']}")

            fert_prices = price_data["prices"].get(fert_prediction, [])
            if fert_prices:
                for item in fert_prices:
                    c1, c2, c3 = st.columns([3, 2, 2])
                    c1.write(f"**{item['brand']}**")
                    c2.write(f"₹{item['price']} / {item['unit']}")
                    c3.markdown(f"[View]({item['link']})")
            else:
                st.info("Price data not available for this fertilizer yet.")

        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")

st.divider()
st.caption("Built as part of SIH25010 — Smart Crop Advisory System for Small and Marginal Farmers")
