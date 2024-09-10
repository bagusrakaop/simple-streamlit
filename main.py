import streamlit as st
from data_analysis import EDA
from inference import HousingPriceModel
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Main App class
class StreamlitApp:
    def __init__(self, data, model_path):
        self.data = data
        self.features = data.drop(columns=['median_house_value']).columns
        self.model = HousingPriceModel(model_path)
        self.eda = EDA(data)
        self.label_enc = self.eda.label_enc
        
    def run(self):
        # Create sidebar for navigation
        st.sidebar.title("Menu")
        options = st.sidebar.radio("Choose an option", ["EDA", "Predict Housing Prices"])
        
        if options == "EDA":
            self.show_eda()
        elif options == "Predict Housing Prices":
            self.predict_prices()
    
    def show_eda(self):
        st.title("Exploratory Data Analysis")
        self.eda.show_head()
        
        # Add checkboxes for more EDA options
        if st.checkbox("Show statistics"):
            self.eda.show_statistics()
        if st.checkbox("Show correlation heatmap"):
            self.eda.plot_corr()
        if st.checkbox("Show pairplot"):
            self.eda.plot_pair()

    def predict_prices(self):
        st.title("Predict Housing Prices")
        
        # Automatically create input fields for each feature
        st.write("## Input the features manually")
        input_data = {}

        for feature in self.features:
            if feature == 'ocean_proximity':
                # Use combo box for ocean_proximity
                unique_values = self.data['ocean_cat'].unique()
                selected_value = st.selectbox(f"{feature}:", unique_values)
                
                # Transform using LabelEncoder
                input_data[feature] = self.label_enc.transform([selected_value])[0]
            elif self.data[feature].dtype == 'float64' or self.data[feature].dtype == 'int64':
                input_data[feature] = st.number_input(f"{feature}:", value=float(self.data[feature].mean()))
            else:
                input_data[feature] = st.text_input(f"{feature}:")

        # Convert input data into a DataFrame
        input_df = pd.DataFrame([input_data])

        if st.button("Predict"):
            prediction = self.model.predict(input_df)
            st.write(f"## Predicted Housing Price: ${prediction[0]:,.2f}")