# app.py
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from utils.clean_text_light import clean_text_light
from utils.preprocess_text import preprocess_text


# -----------------------------
# 1️⃣ Page configuration
# -----------------------------
st.set_page_config(
    page_title="📨 Spam SMS Classifier",
    layout="wide",
)


# -----------------------------
#  🎨 CUSTOM CSS STYLING
# -----------------------------
st.markdown("""
    <style>
        /* Background color for the whole app */
        .stApp {
            background-color: #FFEFD5;  /* Light peach / pale orange */
        }

        /* Sidebar background */
        section[data-testid="stSidebar"] {
            background-color: #E0F0FF !important;  /* Light blue */
        }

        /* Bold title */
        h1 {
            font-weight: 700 !important;
        }
        
        /* Text input area styling */
        textarea {
            background-color: #FFFFFF !important;
            border: 1px solid #CCCCCC !important;
        }

        /* Buttons styling */
        .stButton>button {
            background-color: #4A90E2;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            border: none;
        }

        .stButton>button:hover {
            background-color: #357ABD;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 3️⃣ Title and awareness message
# -----------------------------
st.markdown("# **📨 SMS Spam Classifier**")
st.markdown("Enter your SMS message to find out whether it is SPAM or HAM.")

# Awareness message
st.info(
    "**Why this matters:**\n\n"
    "SMS spam can be more than annoying — it often includes phishing links, "
    "scams, or malicious attempts to steal personal information. Detecting these "
    "messages early helps protect your privacy, finances, and digital security."
)

# -----------------------------
#  4️⃣Load resources
# -----------------------------
@st.cache_resource
def load_tokenizer(path="tokenizer/tokenizer.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_lstm_model(path="model/spam_detector_lstm.h5"):
    return load_model(path)

tokenizer = load_tokenizer()
model = load_lstm_model()

# -----------------------------
# 5️⃣ Sidebar for input
# -----------------------------
with st.sidebar:
    st.header("💬 Enter your message")
    user_input = st.text_area("SMS message:", height=150)
    threshold = st.slider(
        "SPAM threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    predict_button = st.button("Predict")

# -----------------------------
# 6️⃣  Main area
# -----------------------------

if predict_button:
    if user_input.strip() == "":
        st.warning("Please enter a message to predict.")
    else:
        # Preprocessing and padding
        X_pad = preprocess_text([user_input], tokenizer, max_len=40)
        
        # Prediction
        prob = model.predict(X_pad)[0][0]
        pred_label = "SPAM" if prob > threshold else "HAM"

        # Show result with colors
        if pred_label == "SPAM":
            st.error(f"Prediction: {pred_label}")
        else:
            st.success(f"Prediction: {pred_label}")

        # Probability bar
        st.markdown("**Probability of being SPAM:**")
        st.info(f"{prob*100:.2f}%")

        # Original message
        st.markdown("**Evaluated message:**")
        st.write(user_input)


# -----------------------------
# 5️⃣ Footer
# -----------------------------
st.markdown("---")
st.markdown("💡 LSTM-based model trained on SMS spam/ham dataset.")
