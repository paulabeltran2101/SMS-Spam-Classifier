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

st.title("📨 SMS Spam Classifier")
st.markdown("Enter a SMS message below to predict whether it is SPAM or HAM.")

# -----------------------------
# 2️⃣ Load resources
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
# 3️⃣ Sidebar for input
# -----------------------------
with st.sidebar:
    st.header("💬 Enter your message")
    user_input = st.text_area("SMS message:", height=150)
    threshold = st.slider("SPAM threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    predict_button = st.button("Predict")

# -----------------------------
# 4️⃣ Main area
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
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

            # Probability visual
            st.progress(int(prob * 100))
            st.info(f"Probability: {prob*100:.2f}%")

            # Original message
            st.markdown("**Evaluated message:**")
            st.write(user_input)

with col2:
    st.image("images/spam.jng", caption="Spam Detection", use_column_width=True)

# -----------------------------
# 5️⃣ Footer
# -----------------------------
st.markdown("---")
st.markdown("💡 LSTM-based model trained on SMS spam/ham dataset.")
