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
# 2️⃣ Custom CSS 
# -----------------------------
st.markdown("""
<style>
/* --------------------------- */
/* 🔤 GLOBAL FONT (Montserrat) */
/* --------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif !important;
}

/* --------------------------- */
/* 🎨 APP BACKGROUND - image with transparency */
/* --------------------------- */
.stApp {
    /* background-image removed for testing */
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
    position: relative;
}
.stApp::before {
    content: "";
    position: absolute;
    top:0; left:0;
    width: 100%; height: 100%;
    background-color: rgba(255,255,255,0.35);
    z-index: -1;
}

/* --------------------------- */
/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #E0F0FF !important;
}


/* BUTTON STYLING */
.stButton>button {
    background-color: #4A90E2 !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 0.6em 1.4em !important;
    border: none !important;
    font-weight: 600 !important;
    transition: all 0.2s ease-in-out !important;
}
.stButton>button:hover {
    background-color: #357ABD !important;
    transform: translateY(-2px);
    box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
}


/* (alerts) */
div[data-testid="stAlert"] {
    border-radius: 10px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

/* TEXTAREA STYLE */
textarea {
    background-color: #FFFFFF !important;
    border-radius: 8px !important;
    border: 1px solid #BBBBBB !important;
    padding: 10px !important;
}
textarea:focus {
    border-color: #4A90E2 !important;
    box-shadow: 0px 0px 6px rgba(74,144,226,0.5);
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
        st.progress(int(prob * 100))

        # Original message
        st.markdown("**Evaluated message:**")
        st.write(user_input)


# -----------------------------
# 5️⃣ Footer
# -----------------------------
st.markdown("---")
st.markdown("💡 LSTM-based model trained on SMS spam/ham dataset.")
