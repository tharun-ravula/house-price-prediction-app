import gzip
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

MODEL_PATH = Path(__file__).with_name("model.pkl")
DATASET_PATH = Path(__file__).with_name("house_price_prediction_dataset.csv")

with gzip.open(MODEL_PATH, "rb") as pickle_in:
    model = pickle.load(pickle_in)

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "square_feet",
    "lot_size_sqft",
    "year_built",
    "garage_spaces",
    "distance_to_city_center_km",
    "crime_rate_index",
    "nearby_school_rating",
    "parking_spaces",
    "days_on_market",
]

CATEGORICAL_FEATURES = [
    "city",
    "property_type",
    "furnished",
    "property_condition",
    "swimming_pool",
]

MODEL_FEATURE_COLUMNS = [
    "city",
    "property_type",
    "bedrooms",
    "bathrooms",
    "square_feet",
    "lot_size_sqft",
    "year_built",
    "garage_spaces",
    "furnished",
    "property_condition",
    "distance_to_city_center_km",
    "crime_rate_index",
    "nearby_school_rating",
    "swimming_pool",
    "parking_spaces",
    "days_on_market",
]


def load_category_options():
    if not DATASET_PATH.exists():
        return {
            "city": ["Houston", "Austin", "San Diego", "Philadelphia", "Los Angeles"],
            "property_type": ["Condo", "Single Family"],
            "furnished": ["Yes", "No"],
            "property_condition": ["Excellent", "Good", "Fair"],
            "swimming_pool": ["Yes", "No"],
        }

    data = pd.read_csv(DATASET_PATH)
    return {
        "city": sorted(data["city"].dropna().unique().tolist()),
        "property_type": sorted(data["property_type"].dropna().unique().tolist()),
        "furnished": sorted(data["furnished"].dropna().unique().tolist()),
        "property_condition": sorted(data["property_condition"].dropna().unique().tolist()),
        "swimming_pool": sorted(data["swimming_pool"].dropna().unique().tolist()),
    }


def predict(feature_values):
    feature_row = pd.DataFrame([feature_values], columns=MODEL_FEATURE_COLUMNS)
    prediction = model.predict(feature_row)
    return prediction[0]


def get_house_image():
    image_path = Path(__file__).with_name("house.jpg")
    if image_path.exists():
        return Image.open(image_path)

    placeholder = Image.new("RGB", (300, 200), color=(220, 230, 240))
    return placeholder


def main():
    st.title("House Price Prediction")

    template = """
    <div style="background-color:black;padding:10px;font-size:23px;">
        <h1 style="color:white;text-align:center;">House Price Prediction App</h1>
    </div>
    """
    st.markdown(template, unsafe_allow_html=True)
    st.image(get_house_image(), width=300, caption="House-worth")

    category_options = load_category_options()

    feature_values = {
        "bedrooms": st.number_input("Bedrooms", min_value=0, value=3, step=1),
        "bathrooms": st.number_input("Bathrooms", min_value=0, value=2, step=1),
        "square_feet": st.number_input("Square Feet", min_value=0, value=1800, step=50),
        "lot_size_sqft": st.number_input("Lot Size (sqft)", min_value=0, value=4000, step=100),
        "year_built": st.number_input("Year Built", min_value=1900, max_value=2025, value=2000, step=1),
        "garage_spaces": st.number_input("Garage Spaces", min_value=0, value=1, step=1),
        "distance_to_city_center_km": st.number_input("Distance to City Center (km)", min_value=0.0, value=8.0, step=0.5),
        "crime_rate_index": st.number_input("Crime Rate Index", min_value=0.0, value=3.0, step=0.5),
        "nearby_school_rating": st.number_input("Nearby School Rating", min_value=0.0, value=7.0, step=0.1),
        "parking_spaces": st.number_input("Parking Spaces", min_value=0, value=2, step=1),
        "days_on_market": st.number_input("Days on Market", min_value=0, value=30, step=1),
        "city": st.selectbox("City", category_options["city"]),
        "property_type": st.selectbox("Property Type", category_options["property_type"]),
        "furnished": st.selectbox("Furnished", category_options["furnished"]),
        "property_condition": st.selectbox("Property Condition", category_options["property_condition"]),
        "swimming_pool": st.selectbox("Swimming Pool", category_options["swimming_pool"]),
    }

    if st.button("Predict"):
        prediction = predict(feature_values)
        st.success(f"Predicted house price is ${prediction:,.0f} ({prediction / 100000:.2f} lakhs)")


if __name__ == "__main__":
    main()
