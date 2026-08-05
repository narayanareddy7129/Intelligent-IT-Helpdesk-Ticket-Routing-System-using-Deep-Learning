import pickle
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

from utils import preprocess_text

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Intelligent Helpdesk Ticket Classifier",
    page_icon="🎫",
    layout="wide"
)

# ==========================================================
# Load Model & Resources
# ==========================================================

@st.cache_resource
def load_resources():

    model = load_model("gru_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_resources()

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("🎫 Project Information")

st.sidebar.markdown("""
## Model

- **Architecture:** GRU
- **Framework:** TensorFlow
- **Embedding Dimension:** 128
- **Max Sequence Length:** 300

---

## Preprocessing

- Lowercase
- Contraction Expansion
- Remove Special Characters
- Tokenization
- Padding

---

## Deep Learning Pipeline

Text

⬇

Preprocessing

⬇

Tokenizer

⬇

Embedding

⬇

GRU

⬇

Dense

⬇

Softmax
""")

# ==========================================================
# Main Title
# ==========================================================

st.title("🎫 Intelligent Helpdesk Ticket Classifier")

st.markdown("""
This application predicts the **department/category**
for customer support tickets using a **GRU-based Deep Learning model**.
""")

st.divider()

# ==========================================================
# Sample Tickets
# ==========================================================

# sample_ticket = st.selectbox(
#     "Choose a Sample Ticket (Optional)",
#     [
#         "None",
#         "I cannot login to my account after resetting my password.",
#         "My internet connection is very slow and keeps disconnecting.",
#         "I was charged twice for my subscription.",
#         "My laptop is overheating during normal usage.",
#         "The VPN connection is failing every morning."
#     ]
# )

default_text = ""

# if sample_ticket != "None":
#     default_text = sample_ticket

# ==========================================================
# User Input
# ==========================================================

ticket = st.text_area(
    "Enter Support Ticket",
    value=default_text,
    height=220,
    placeholder="Describe your issue..."
)

# ==========================================================
# Prediction
# ==========================================================

if st.button("🚀 Predict Category", use_container_width=True):

    if ticket.strip() == "":
        st.warning("Please enter a support ticket.")
        st.stop()

    cleaned_text, processed_text = preprocess_text(
        ticket,
        tokenizer
    )

    prediction = model.predict(
        processed_text,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    confidence = float(
        prediction[0][predicted_index]
    )

    predicted_category = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    st.divider()

    # ======================================================
    # Prediction Result
    # ======================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎯 Prediction")

        st.success(predicted_category)

    with col2:

        st.subheader("Confidence")

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2%}"
        )

        st.progress(confidence)

    # ======================================================
    # Top 3 Predictions
    # ======================================================

    st.subheader("🏆 Top 3 Predictions")

    probs = prediction[0]

    top3 = np.argsort(probs)[::-1][:3]

    top3_df = pd.DataFrame({
        "Category": label_encoder.inverse_transform(top3),
        "Confidence": [f"{probs[i]:.2%}" for i in top3]
    })

    st.table(top3_df)

    # ======================================================
    # Probability Chart
    # ======================================================

    st.subheader("📊 Prediction Probabilities")

    probability_df = pd.DataFrame({
        "Category": label_encoder.classes_,
        "Probability": probs
    })

    st.bar_chart(
        probability_df.set_index("Category")
    )

    # ======================================================
    # Cleaned Text
    # ======================================================

    with st.expander("📝 View Cleaned Text"):

        st.code(cleaned_text)

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "Developed using TensorFlow • GRU • NLP • Streamlit"
)