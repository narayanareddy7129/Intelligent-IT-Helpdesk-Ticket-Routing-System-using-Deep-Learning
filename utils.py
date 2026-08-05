import re
import contractions
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Same value used during training
MAX_LEN = 300


def dl_clean_text(text):
    """
    Preprocess text exactly as done during training.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = contractions.fix(text)
    text = text.replace("\\n", " ")
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text



def preprocess_text(text, tokenizer):

    cleaned = dl_clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    return cleaned, padded