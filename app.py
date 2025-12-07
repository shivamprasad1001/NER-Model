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

# Color palette (defined early for use throughout)
COLOR_PALETTE = [
    "#d8e2dc", "#ffe5d9", "#ffcad4", "#f4acb7", "#9d8189",
    "#e8e8e4", "#d5bdaf", "#cdb4db", "#ffc8dd", "#ffafcc"
]

COLOR_MAP = {label: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, label in enumerate(ENTITY_LABELS)}

# ================================
# Page Configuration
# ================================
st.set_page_config(
    page_title="NER Model Inference",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 16px;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .entity-label {
        display: inline-block;
        margin: 0.25rem;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    h1 {
        color: #1f2937;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    h2, h3 {
        color: #374151;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    .highlighted-output {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        line-height: 1.8;
        font-size: 16px;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# ================================
# Sidebar Information
# ================================
st.sidebar.markdown("### 🤖 Model Information")
st.sidebar.info(f"**Model Path:** `{MODEL_DIR}`")

st.sidebar.markdown("### 🏷️ Entity Labels")
if ENTITY_LABELS:
    labels_html = "".join([
        f'<span class="entity-label" style="background-color: {COLOR_PALETTE[i % len(COLOR_PALETTE)]}; color: #1f2937;">{label}</span>'
        for i, label in enumerate(ENTITY_LABELS)
    ])
    st.sidebar.markdown(f'<div style="margin-top: 0.5rem;">{labels_html}</div>', unsafe_allow_html=True)
else:
    st.sidebar.warning("No labels found.")

with st.sidebar.expander("ℹ️ About This Model", expanded=False):
    st.markdown("""
    This model was trained using a custom NER pipeline from scratch. It supports 
    Doccano-style annotations and extracts structured entities from natural language text.
    
    **Training Focus:**
    - Desktop control commands
    - System operations
    - Reminders & scheduling
    - Application management
    - WiFi connectivity
    - Music playback
    - And more...
    """)

with st.sidebar.expander("📖 How to Use", expanded=False):
    st.markdown("""
    1. **Enter text** in the input field below
    2. **Click** 'Extract Entities' button
    3. **Review** highlighted entities and results table
    4. **Experiment** with different inputs!
    """)

# ================================
# Main Content
# ================================
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🔍 Named Entity Recognition")
    st.markdown("Extract structured information from natural language text")
with col2:
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    if ENTITY_LABELS:
        st.metric("Entity Types", len(ENTITY_LABELS))

st.markdown("---")

# Dataset Info in expandable section
with st.expander("📊 About the Training Dataset", expanded=False):
    st.markdown("""
    This model is trained on a custom dataset of natural language commands covering:
    
    **Command Categories:**
    - Desktop control & system operations
    - Reminders & task management
    - Application management
    - Network connectivity
    - Music & media playback
    - Profile updates & weather queries
    
    **Dataset Characteristics:**
    - 40+ entity categories (device, location, app_name, artist, date, time, etc.)
    - JSONL format with annotated spans
    - Variety-focused (conversational + imperative styles)
    - Suitable for experimentation and learning
    
    ⚠️ **Note:** This model is designed for study purposes and is not production-ready.
    """)

# Main input section
st.markdown("### 💬 Test Your Input")
st.markdown('<div class="info-box">Enter any text below to see how the model identifies and extracts entities.</div>', unsafe_allow_html=True)

text = st.text_area(
    "Input Text",
    height=140,
    placeholder="Example: Set brightness to 80% and play some jazz music by Miles Davis",
    label_visibility="collapsed"
)

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
                padding:3px 8px;
                border-radius:5px;
                font-weight:500;
                font-size:15px;
                margin: 0 2px;
                display: inline-block;
            ">
            {text[start:end]} <span style="font-size:11px; opacity:0.8;">({label})</span>
            </span>
        """
        last_end = end

    html += text[last_end:]
    return html

# ================================
# Run Inference
# ================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    extract_button = st.button("🚀 Extract Entities", use_container_width=True)

if extract_button:
    if not text.strip():
        st.warning("⚠️ Please enter some text before running inference.")
    else:
        with st.spinner("Analyzing text..."):
            doc = nlp(text)

            entities = [
                {"text": ent.text, "start": ent.start_char, "end": ent.end_char, "label": ent.label_}
                for ent in doc.ents
            ]

        st.markdown("---")
        
        # Results summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Input Length", f"{len(text)} chars")
        with col2:
            st.metric("✨ Entities Found", len(entities))
        with col3:
            unique_labels = len(set(ent["label"] for ent in entities)) if entities else 0
            st.metric("🏷️ Unique Types", unique_labels)

        st.markdown("<br>", unsafe_allow_html=True)

        # Highlighted text
        st.markdown("### 🎨 Highlighted Text")
        highlighted_html = highlight_text(text, entities)
        st.markdown(
            f"<div class='highlighted-output'>{highlighted_html}</div>",
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Extracted entities table
        st.markdown("### 📋 Extracted Entities")
        if entities:
            df = pd.DataFrame(entities)
            # Reorder columns for better presentation
            df = df[["label", "text", "start", "end"]]
            df.columns = ["Entity Type", "Text", "Start", "End"]
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("ℹ️ No entities detected in this text. Try a different input!")