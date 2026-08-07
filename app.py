import pickle

import numpy as np
import pandas as pd
import streamlit as st
import torch
from tensorflow.keras.models import load_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
# Paths — adjust these if your folder layout differs
# ==========================================================

TOKENIZER_PATH = "tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

RNN_MODEL_PATHS = {
    "GRU": "models/gru_model.keras",
    "LSTM": "models/lstm_model.keras",
    "BiLSTM": "models/bilstm_model.keras",
}

BERT_MODEL_DIR = "bert_ticket_classifier"

# Max length used when tokenizing text for BERT. This wasn't specified
# in config.json (that only stores max_position_embeddings=512, the
# model's absolute ceiling) — 128 is a common default for short ticket
# text. If BERT was fine-tuned with a different max_length, change this
# to match, or predictions may be less accurate on long tickets.
BERT_MAX_LEN = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# Load Resources
# ==========================================================

@st.cache_resource
def load_shared_resources():
    """Keras tokenizer + label encoder, shared by GRU/LSTM/BiLSTM and
    used to decode BERT's predicted class index back to a category name."""

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return tokenizer, label_encoder


@st.cache_resource
def load_rnn_model(model_name):
    return load_model(RNN_MODEL_PATHS[model_name])


@st.cache_resource
def load_bert_resources():
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    bert_model.to(DEVICE)
    bert_model.eval()
    return bert_tokenizer, bert_model


keras_tokenizer, label_encoder = load_shared_resources()

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("🎫 Project Information")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["GRU", "LSTM", "BiLSTM", "BERT"]
)

ARCH_INFO = {
    "GRU": """
## Model

- **Architecture:** GRU
- **Framework:** TensorFlow / Keras
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
""",
    "LSTM": """
## Model

- **Architecture:** LSTM
- **Framework:** TensorFlow / Keras
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
LSTM
⬇
Dense
⬇
Softmax
""",
    "BiLSTM": """
## Model

- **Architecture:** Bidirectional LSTM
- **Framework:** TensorFlow / Keras
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
Bidirectional LSTM
⬇
Dense
⬇
Softmax
""",
    "BERT": f"""
## Model

- **Architecture:** BERT (bert-base-uncased)
- **Framework:** Hugging Face Transformers (PyTorch)
- **Hidden Size:** 768
- **Layers / Heads:** 12 / 12
- **Max Sequence Length:** {BERT_MAX_LEN}

---

## Preprocessing

- WordPiece Tokenization
- Truncation / Padding
- Attention Mask

---

## Deep Learning Pipeline

Text
⬇
WordPiece Tokenizer
⬇
BERT Encoder (12 layers)
⬇
[CLS] Pooled Output
⬇
Dense
⬇
Softmax
""",
}

st.sidebar.markdown(ARCH_INFO[model_choice])

# ==========================================================
# Main Title
# ==========================================================

st.title("🎫 Intelligent Helpdesk Ticket Classifier")

st.markdown("""
This application predicts the **department/category**
for customer support tickets using your choice of a
**GRU, LSTM, BiLSTM, or BERT** model.
""")

st.divider()

# ==========================================================
# User Input
# ==========================================================

ticket = st.text_area(
    "Enter Support Ticket",
    value="",
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

    if model_choice == "BERT":

        with st.spinner("Loading BERT model..."):
            bert_tokenizer, bert_model = load_bert_resources()

        display_text = ticket.strip()

        encoded = bert_tokenizer(
            display_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=BERT_MAX_LEN,
        ).to(DEVICE)

        with torch.no_grad():
            logits = bert_model(**encoded).logits

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    else:

        with st.spinner(f"Loading {model_choice} model..."):
            rnn_model = load_rnn_model(model_choice)

        display_text, processed_text = preprocess_text(
            ticket,
            keras_tokenizer
        )

        prediction = rnn_model.predict(
            processed_text,
            verbose=0
        )

        probs = prediction[0]

    predicted_index = int(np.argmax(probs))

    confidence = float(probs[predicted_index])

    # NOTE: assumes BERT was fine-tuned on the same label-encoded
    # targets as the RNN models — config.json only stores generic
    # LABEL_0..LABEL_7, so real category names come from label_encoder.
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
    # Processed Text
    # ======================================================

    expander_label = (
        "📝 View Cleaned Text" if model_choice != "BERT"
        else "📝 View Text Sent to Model"
    )

    with st.expander(expander_label):

        st.code(display_text)

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "Developed using TensorFlow • PyTorch • Transformers • NLP • Streamlit"
)
