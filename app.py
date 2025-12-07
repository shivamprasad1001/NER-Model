import streamlit as st
import spacy
import pickle
from pathlib import Path
import pandas as pd

# ================================
# Load Model
# ================================
MODEL_DIR = "./simple_ner_model"

try:
    nlp = spacy.load(MODEL_DIR)
except Exception as e:
    st.error(f"Failed to load model from {MODEL_DIR}: {e}")
    st.stop()

# Load entity labels
label_path = Path(MODEL_DIR) / "entity_labels.pkl"
if label_path.exists():
    ENTITY_LABELS = pickle.load(open(label_path, "rb"))
else:
    ENTITY_LABELS = []

# ================================
# Page Configuration
# ================================
st.set_page_config(
    page_title="NER Model Inference",
    layout="wide"
)

# ================================
# Sidebar Information
# ================================
st.sidebar.title("Model Information")
st.sidebar.write("This interface runs inference using your custom-trained Named Entity Recognition model.")

st.sidebar.subheader("Model Path")
st.sidebar.write(MODEL_DIR)

st.sidebar.subheader("Entity Labels")
if ENTITY_LABELS:
    st.sidebar.write(", ".join(ENTITY_LABELS))
else:
    st.sidebar.write("No labels found.")

st.sidebar.subheader("Purpose")
st.sidebar.write(
    """
    This model was trained using a custom NER training pipeline built from scratch 
    without spaCy lookup dependencies.  
    It supports Doccano-style annotated text and extracts entities from user input.
    """
)

st.sidebar.subheader("Usage Instructions")
st.sidebar.write(
    """
    1. Enter text in the provided input field.  
    2. Click 'Extract Entities'.  
    3. Review highlighted entities and the results table.
    """
)

# ================================
# Main Content
# ================================
st.title("Custom NER Model Inference")

st.write(
    """
    This page allows you to test your trained NER model on any input text. 
    The output includes a highlighted version of the text and a structured table of detected entities.
    """
)

text = st.text_area(
    "Input Text",
    height=160,
    placeholder="Type any sentence to run NER inference."
)

# Subtle, neutral colors for labels
COLOR_PALETTE = [
    "#d8e2dc", "#ffe5d9", "#ffcad4", "#f4acb7", "#9d8189",
    "#e8e8e4", "#d5bdaf", "#cdb4db", "#ffc8dd", "#ffafcc"
]

COLOR_MAP = {label: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, label in enumerate(ENTITY_LABELS)}

# ================================
# Entity Highlighting Utility
# ================================
def highlight_text(text, entities):
    if not entities:
        return text

    entities = sorted(entities, key=lambda x: x["start"])
    html = ""
    last_end = 0

    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        label = ent["label"]
        color = COLOR_MAP.get(label, "#e8e8e8")

        html += text[last_end:start]
        html += f"""
            <span style="
                background-color:{color};
                padding:2px 4px;
                border-radius:3px;
                font-size:14px;
            ">
            {text[start:end]} ({label})
            </span>
        """
        last_end = end

    html += text[last_end:]
    return html

# ================================
# Run Inference
# ================================
if st.button("Extract Entities"):
    if not text.strip():
        st.warning("Please enter text before running inference.")
    else:
        doc = nlp(text)

        entities = [
            {"text": ent.text, "start": ent.start_char, "end": ent.end_char, "label": ent.label_}
            for ent in doc.ents
        ]

        st.subheader("Highlighted Text")
        st.markdown(
            f"<div style='font-size:17px; line-height:1.6;'>{highlight_text(text, entities)}</div>",
            unsafe_allow_html=True
        )

        st.subheader("Extracted Entities")
        if entities:
            st.table(pd.DataFrame(entities))
        else:
            st.info("No entities detected in this text.")
