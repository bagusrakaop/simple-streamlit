import streamlit as st
from catboost import CatBoostRegressor

# Prediction Model class using pre-trained CatBoost model
class HousingPriceModel:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        # st.write("## Loading Pre-Trained Model")
        model = CatBoostRegressor()
        model = model.load_model(model_path)
        # st.write("Model loaded successfully!")
        return model

    def predict(self, input_data):
        st.write("## Predicting Housing Price")
        prediction = self.model.predict(input_data)
        return prediction
