import os

import streamlit as st
from dotenv import load_dotenv
from google import genai


# ==========================================================
# Load environment variables
# ==========================================================

load_dotenv()


# ==========================================================
# Ticket Categories
# ==========================================================

CATEGORIES = [
    "Hardware",
    "HR Support",
    "Access",
    "Miscellaneous",
    "Storage",
    "Purchase",
    "Internal Project",
    "Administrative rights"
]


# ==========================================================
# Get Gemini API Key
# ==========================================================

def get_gemini_api_key():

    # Local environment
    api_key = os.getenv("GEMINI_API_KEY")

    # Streamlit Cloud
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Set it in .env for local use or "
            "Streamlit Secrets for deployment."
        )

    return api_key


# ==========================================================
# Gemini Client
# ==========================================================

client = genai.Client(
    api_key=get_gemini_api_key()
)


# ==========================================================
# Gemini Prediction
# ==========================================================

def gemini_predict(ticket):

    prompt = f"""
You are an IT helpdesk ticket classification system.

Your task is to classify the following IT support ticket
into exactly ONE of these categories:

{", ".join(CATEGORIES)}

Rules:

1. Return exactly one category from the list.
2. Do not create a new category.
3. Do not provide explanations.
4. Return only the category name.
5. Match the category name exactly.

Ticket:

{ticket}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    prediction = response.text.strip()

    # ======================================================
    # Validate Gemini Output
    # ======================================================

    if prediction not in CATEGORIES:
        prediction = "Miscellaneous"

    return prediction