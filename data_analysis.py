import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Data Processing and EDA class
class EDA:
    def __init__(self, data):
        self.data = data
        self.data['ocean_cat'] = self.data['ocean_proximity']
        self.label_enc = LabelEncoder()
        self.data['ocean_proximity'] = self.label_enc.fit_transform(self.data['ocean_proximity'])
    
    def show_head(self):
        st.write("## Dataset Overview")
        st.dataframe(self.data.head())
    
    def show_statistics(self):
        st.write("## Data Statistics")
        st.write(self.data.describe())
    
    def plot_corr(self):
        st.write("## Correlation Heatmap")
        corr = self.data.drop(['ocean_cat'], axis=1).corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot(plt)
    
    def plot_pair(self):
        st.write("## Pairplot")
        plt.figure(figsize=(12, 8))
        sns.pairplot(self.data.drop(['ocean_cat'], axis=1))
        st.pyplot(plt)