import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load('model.pkl')

# Set page config
st.set_page_config(
    page_title="Air Pollution Monitoring System",
    page_icon="💨",
    layout="wide"
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .stButton>button {
        color: white !important;
        background-color: #007bff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("About")
with st.sidebar:
    st.write(
        """
        Air Pollution Monitoring System

        This application uses machine learning to predict PM2.5 levels based on environmental parameters.

        Parameters:
        """
    )
    st.write("- AOD (Aerosol Optical Depth)")
    st.write("- Temperature (K)")
    st.write("- Humidity (%)")
    st.write("- Wind Speed (m/s)")
    st.write("- PBLH (Planetary Boundary Layer Height, m)")

# Main content
st.title("Air Pollution Monitoring System")

# Input section
col1, col2 = st.columns(2)

with col1:
    aod = st.number_input(
        "AOD",
        min_value=0.0,
        max_value=1.5,
        value=0.5,
        step=0.1
    )
    temperature = st.number_input(
        "Temperature (K)",
        min_value=280.0,
        max_value=320.0,
        value=300.0,
        step=1.0
    )
    humidity = st.number_input(
        "Humidity (%)",
        min_value=30.0,
        max_value=90.0,
        value=50.0,
        step=1.0
    )

with col2:
    windspeed = st.number_input(
        "Wind Speed (m/s)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1
    )
    pblh = st.number_input(
        "PBLH (m)",
        min_value=500.0,
        max_value=2000.0,
        value=1000.0,
        step=100.0
    )

# Prediction button
if st.button("Predict PM2.5"):
    # Prepare input data
    input_data = np.array([[aod, temperature, humidity, windspeed, pblh]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display result
    st.success(f"Predicted PM2.5: {prediction:.2f} μg/m³")

# Add some visualizations
st.header("Model Performance")

# Create some sample data for visualization
np.random.seed(42)
sample_data = {
    'AOD': np.random.uniform(0.1, 1.5, 100),
    'Temperature': np.random.uniform(280, 320, 100),
    'Humidity': np.random.uniform(30, 90, 100),
    'WindSpeed': np.random.uniform(0.5, 5.0, 100),
    'PBLH': np.random.uniform(500, 2000, 100),
}
sample_df = pd.DataFrame(sample_data)
sample_df['PM2.5'] = model.predict(sample_df)

# Correlation heatmap
st.subheader("Correlation Heatmap")
corr = sample_df.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Feature importance
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': ['AOD', 'Temperature', 'Humidity', 'WindSpeed', 'PBLH'],
    'Importance': model.feature_importances_
})
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance.sort_values('Importance', ascending=False),
    ax=ax
)
st.pyplot(fig)
