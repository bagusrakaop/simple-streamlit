import streamlit as st
import pandas as pd
from main import StreamlitApp

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('data/housing.csv') 

# Load the pre-trained CatBoost model
@st.cache_resource
def load_model():
    model_path = 'model/cb_model.cb'  # Path to the CatBoost model
    return model_path

# Run the Streamlit app
if __name__ == "__main__":
    data = load_data()
    model_path = load_model()
    app = StreamlitApp(data, model_path)
    app.run()