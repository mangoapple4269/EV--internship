import streamlit as st

st.title("Hello Streamlit from Colab!")

# Paste your full Streamlit + chatbot + prediction code here
import streamlit as st
import pandas as pd
import joblib
import requests

# ---------------------------
# Chatbot using ApiFreeLLM
# ---------------------------
API_URL = "https://apifreellm.com/api/chat"

def ask_llm(question: str, history: str = "") -> str:
    prompt = history + "\nUser: " + question + "\nAssistant:"
    payload = {"message": prompt}
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(API_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            return f"LLM API error: {resp.status_code} {resp.text}"
        result = resp.json()
        if result.get("status") == "success":
            return result.get("response", "")
        else:
            return f"Error from API: {result.get('error', '')}"
    except Exception as e:
        return f"Error calling ApiFreeLLM: {e}"

# ---------------------------
# Load your Random Forest model
# Replace `ev_model.pkl` with your actual file name
# ---------------------------
@st.cache_resource
def load_model(path="ev_model.pkl"):
    return joblib.load(path)

model = load_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="EV Battery Charging Time + Chatbot", layout="wide")
st.title("EV Battery Charging Time Predictor + Chatbot")

mode = st.radio("Choose mode:", ("Predict Charging Time", "Chatbot"))

if mode == "Predict Charging Time":
    st.subheader("Enter input for prediction")
    battery_size = st.number_input("Battery Size (kWh)", min_value=0.0, step=0.1)
    ev_model = st.text_input("EV Model (exact string used during training)")

    if st.button("Predict"):
        try:
            X_input = pd.DataFrame({'Battery Size': [battery_size], 'EV Model': [ev_model]})
            # If your model required encoding for 'EV Model', replicate it here before prediction
            pred = model.predict(X_input)[0]
            st.write(f"**Predicted Charging Time:** {pred:.2f} hours")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

elif mode == "Chatbot":
    st.subheader("Ask the EV Chatbot")
    user_question = st.text_input("Your question here")
    if st.button("Send"):
        with st.spinner("Getting response... (may take few seconds)"):
            answer = ask_llm(user_question)
        st.write("**You:**", user_question)
        st.write("**Assistant:**", answer)
        st.info("Powered by ApiFreeLLM (free LLM, no API key required)")

