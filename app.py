import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import torch
from tensorflow.keras.models import load_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import preprocess_text
from gemini_model import gemini_predict


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Intelligent Helpdesk Ticket Classifier",
    page_icon="🎫",
    layout="wide"
)


# ==========================================================
# Paths
# ==========================================================

TOKENIZER_PATH = "tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

RNN_MODEL_PATHS = {
    "GRU": "models/gru_model.keras",
    "LSTM": "models/lstm_model.keras",
    "BiLSTM": "models/bilstm_model.keras",
}

BERT_MODEL_DIR = "bert_ticket_classifier"

BERT_MAX_LEN = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# Load Resources
# ==========================================================

@st.cache_resource
def load_shared_resources():

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return tokenizer, label_encoder


@st.cache_resource
def load_rnn_model(model_name):

    return load_model(
        RNN_MODEL_PATHS[model_name]
    )


@st.cache_resource
def load_bert_resources():

    bert_tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_DIR
    )

    bert_model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_DIR
    )

    bert_model.to(DEVICE)
    bert_model.eval()

    return bert_tokenizer, bert_model


# Load shared resources
keras_tokenizer, label_encoder = load_shared_resources()


# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("🎫 Project Information")


model_choice = st.sidebar.selectbox(
    "Choose Model",
    [
        "GRU",
        "LSTM",
        "BiLSTM",
        "BERT",
        "Gemini LLM"
    ]
)


# ==========================================================
# Architecture Information
# ==========================================================

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

    "Gemini LLM": """

## Model

- **Architecture:** Generative LLM
- **Model:** Gemini
- **Type:** Instruction-based classification
- **Method:** Prompt-based classification

---

## Preprocessing

- Original ticket text
- No traditional tokenization
- Ticket is provided directly to the LLM

---

## Generative AI Pipeline

Text
⬇
Classification Prompt
⬇
Gemini LLM
⬇
Category
"""
}


st.sidebar.markdown(
    ARCH_INFO[model_choice]
)


# ==========================================================
# Main Title
# ==========================================================

st.title(
    "🎫 Intelligent Helpdesk Ticket Classifier"
)

st.markdown(
    """
This application predicts the **department/category**
for IT support tickets using **GRU, LSTM, BiLSTM, BERT,
or Gemini LLM**.
"""
)

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

if st.button(
    "🚀 Predict Category",
    use_container_width=True
):

    if ticket.strip() == "":
        st.warning(
            "Please enter a support ticket."
        )
        st.stop()


    # ======================================================
    # Gemini LLM
    # ======================================================

    if model_choice == "Gemini LLM":

        with st.spinner(
            "Gemini is analyzing the ticket..."
        ):

            predicted_category = gemini_predict(
                ticket
            )

        st.divider()

        st.subheader("🎯 Prediction")

        st.success(
            predicted_category
        )

        st.info(
            "Prediction generated using Gemini LLM."
        )

        st.subheader("🤖 Model")

        st.write(
            "Gemini LLM"
        )

        st.subheader(
            "📝 Ticket Sent to Gemini"
        )

        with st.expander(
            "View Ticket"
        ):

            st.code(
                ticket.strip()
            )


    # ======================================================
    # BERT
    # ======================================================

    elif model_choice == "BERT":

        with st.spinner(
            "Loading BERT model..."
        ):

            bert_tokenizer, bert_model = (
                load_bert_resources()
            )


        display_text = ticket.strip()


        encoded = bert_tokenizer(
            display_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=BERT_MAX_LEN,
        )


        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }


        with torch.no_grad():

            logits = bert_model(
                **encoded
            ).logits


        probs = (
            torch.softmax(
                logits,
                dim=1
            )
            .cpu()
            .numpy()[0]
        )


        predicted_index = int(
            np.argmax(probs)
        )


        confidence = float(
            probs[predicted_index]
        )


        predicted_category = (
            label_encoder.inverse_transform(
                [predicted_index]
            )[0]
        )


        # ==================================================
        # Prediction Result
        # ==================================================

        st.divider()

        col1, col2 = st.columns(2)


        with col1:

            st.subheader(
                "🎯 Prediction"
            )

            st.success(
                predicted_category
            )


        with col2:

            st.subheader(
                "Confidence"
            )

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2%}"
            )

            st.progress(
                confidence
            )


        # ==================================================
        # Top 3 Predictions
        # ==================================================

        st.subheader(
            "🏆 Top 3 Predictions"
        )


        top3 = np.argsort(
            probs
        )[::-1][:3]


        top3_df = pd.DataFrame({

            "Category":
                label_encoder.inverse_transform(
                    top3
                ),

            "Confidence":
                [
                    f"{probs[i]:.2%}"
                    for i in top3
                ]
        })


        st.table(
            top3_df
        )


        # ==================================================
        # Probability Chart
        # ==================================================

        st.subheader(
            "📊 Prediction Probabilities"
        )


        probability_df = pd.DataFrame({

            "Category":
                label_encoder.classes_,

            "Probability":
                probs
        })


        st.bar_chart(
            probability_df.set_index(
                "Category"
            )
        )


        # ==================================================
        # Text
        # ==================================================

        with st.expander(
            "📝 View Text Sent to Model"
        ):

            st.code(
                display_text
            )


    # ======================================================
    # GRU / LSTM / BiLSTM
    # ======================================================

    else:

        with st.spinner(
            f"Loading {model_choice} model..."
        ):

            rnn_model = load_rnn_model(
                model_choice
            )


        display_text, processed_text = (
            preprocess_text(
                ticket,
                keras_tokenizer
            )
        )


        prediction = rnn_model.predict(
            processed_text,
            verbose=0
        )


        probs = prediction[0]


        predicted_index = int(
            np.argmax(probs)
        )


        confidence = float(
            probs[predicted_index]
        )


        predicted_category = (
            label_encoder.inverse_transform(
                [predicted_index]
            )[0]
        )


        # ==================================================
        # Prediction Result
        # ==================================================

        st.divider()


        col1, col2 = st.columns(2)


        with col1:

            st.subheader(
                "🎯 Prediction"
            )

            st.success(
                predicted_category
            )


        with col2:

            st.subheader(
                "Confidence"
            )

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2%}"
            )

            st.progress(
                confidence
            )


        # ==================================================
        # Top 3 Predictions
        # ==================================================

        st.subheader(
            "🏆 Top 3 Predictions"
        )


        top3 = np.argsort(
            probs
        )[::-1][:3]


        top3_df = pd.DataFrame({

            "Category":
                label_encoder.inverse_transform(
                    top3
                ),

            "Confidence":
                [
                    f"{probs[i]:.2%}"
                    for i in top3
                ]
        })


        st.table(
            top3_df
        )


        # ==================================================
        # Probability Chart
        # ==================================================

        st.subheader(
            "📊 Prediction Probabilities"
        )


        probability_df = pd.DataFrame({

            "Category":
                label_encoder.classes_,

            "Probability":
                probs
        })


        st.bar_chart(
            probability_df.set_index(
                "Category"
            )
        )


        # ==================================================
        # Processed Text
        # ==================================================

        with st.expander(
            "📝 View Cleaned Text"
        ):

            st.code(
                display_text
            )


# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "Developed using TensorFlow • PyTorch • Transformers • "
    "Gemini • NLP • Streamlit"
)