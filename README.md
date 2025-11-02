EV Battery Health Predictor

This project predicts EV battery health and classifies optimal charging duration using a Random Forest model. Includes a Streamlit app and a simple chatbot powered by a free LLM API.


Files:

app.py – Streamlit frontend + chatbot integration
ev_model.pkl – trained Random Forest model
requirement.txt – dependencies
notebooks/EvBattery.ipynb – EDA and preprocessing


Dataset:
Contains 1000 entries and 13 columns:

SOC (%), Voltage (V), Current (A), Battery Temp (°C), Ambient Temp (°C), Charging Duration (min), Degradation Rate (%), Charging Mode, Efficiency (%), Battery Type, Charging Cycles, EV Model, Optimal Charging Duration Class


Dependencies:
streamlit, pandas, joblib, requests, scikit-learn


Run the App:

pip install -r requirements.txt
streamlit run app.py


Note: Some features may not work fully on Streamlit Cloud.
