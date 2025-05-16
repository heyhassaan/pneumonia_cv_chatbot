import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
import requests
import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Page configuration with custom styling
st.set_page_config(
    page_title="PneumoAssist: X-Ray Analysis & Healthcare Assistant",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS for refined UI
st.markdown("""
<style>
    body, .stApp {
        background-color: #f9fbff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333333;
    }
    .main {
        padding: 1rem 2rem;
        max-width: 1100px;
        margin: auto;
    }
    .header-section {
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        color: #2c3e50;
        letter-spacing: 1px;
    }
    .info-box {
        background-color: #e7f0ff;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgb(0 0 0 / 0.05);
        font-size: 14px;
        line-height: 1.5;
        color: #1f4e79;
    }
    .result-box {
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        box-shadow: 0 2px 8px rgb(0 0 0 / 0.1);
        font-size: 15px;
        line-height: 1.6;
    }
    .result-normal {
        background-color: #d9f3db;
        border-left: 6px solid #3b9d51;
        color: #2d5a29;
    }
    .result-pneumonia {
        background-color: #ffe6e6;
        border-left: 6px solid #cc3333;
        color: #7a2424;
    }
    .disclaimer {
        font-size: 12px;
        color: #666;
        font-style: italic;
        margin-top: 12px;
    }
    .chat-container {
        border: 1px solid #cbd6e2;
        border-radius: 12px;
        padding: 20px;
        background-color: #ffffff;
        height: 420px;
        overflow-y: auto;
        box-shadow: inset 0 0 10px rgb(0 0 0 / 0.03);
    }
    .user-message {
        background-color: #d9e8ff;
        padding: 12px 18px;
        border-radius: 18px 18px 0 18px;
        margin: 10px 0;
        max-width: 75%;
        float: right;
        clear: both;
        color: #1a3e72;
        font-weight: 500;
    }
    .assistant-message {
        background-color: #f3f6fb;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 0;
        margin: 10px 0;
        max-width: 75%;
        float: left;
        clear: both;
        color: #2f3e4d;
    }
    input[type="text"] {
        padding: 12px 15px;
        border-radius: 25px;
        border: 1.5px solid #b3c7e6;
        width: 100%;
        font-size: 15px;
        transition: border-color 0.3s ease;
    }
    input[type="text"]:focus {
        border-color: #3b78e7;
        outline: none;
        box-shadow: 0 0 8px rgba(59, 120, 231, 0.3);
    }
    button[type="submit"] {
        background-color: #3b78e7;
        color: white;
        border: none;
        padding: 12px 30px;
        font-size: 15px;
        border-radius: 25px;
        cursor: pointer;
        transition: background-color 0.25s ease;
    }
    button[type="submit"]:hover {
        background-color: #315fbb;
    }
    /* Clear floats after chat messages */
    .chat-container::after {
        content: "";
        clear: both;
        display: table;
    }
    .footer {
        text-align: center;
        margin-top: 35px;
        padding: 12px;
        background-color: #eef2f8;
        border-radius: 10px;
        font-size: 13px;
        color: #666666;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='header-section'><h1>🫁 PneumoAssist: X-Ray Analysis & Healthcare Assistant</h1></div>", unsafe_allow_html=True)

# Cache the model load
@st.cache_resource
def load_pneumonia_model():
    try:
        return load_model('cnn_model91.h5')
    except Exception as e:
        st.error(f"Error loading pneumonia model: {e}")
        return None

pneumonia_model = load_pneumonia_model()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Hello! I'm PneumoAssist, your healthcare assistant focused on chest X-ray "
            "analysis for pneumonia detection. I provide objective information and guidance "
            "based strictly on uploaded images and general knowledge. "
            "I do NOT provide diagnosis or medical advice beyond educational content. "
            "How can I assist you today?"
        )
    }]

def process_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        display_image = image.copy()
        image = image.convert('L')
        image = image.resize((150, 150))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array, display_image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def chat_response(user_input):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

    # Restrict chat to safe, informational responses about pneumonia and x-ray analysis
    safe_system_prompt = """<s>[INST] <<SYS>>
You are PneumoAssist, a concise and cautious healthcare assistant. You only provide
educational information on pneumonia and X-ray analysis based strictly on uploaded
images and general guidelines. Do NOT offer diagnosis, prognosis, or symptom
assessment. Avoid suggesting medical conditions or treatments. Always
recommend consulting qualified healthcare professionals for any medical concerns.
Respond briefly and clearly to user queries within these boundaries.
<</SYS>>"""

    history = st.session_state.messages[-5:-1] if len(st.session_state.messages) > 1 else []
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

    prompt = f"""{safe_system_prompt}

{history_str}
User: {user_input} [/INST]"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.1,
            "do_sample": False,
            "repetition_penalty": 1.5
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        full_text = response.json()[0]['generated_text']
        answer = full_text.split("[/INST]")[-1].strip()
        # Extra safety: Remove any lines suggesting diagnosis or medical advice beyond the prompt
        filtered_answer = "\n".join(
            line for line in answer.split("\n")
            if not any(
                banned_word in line.lower()
                for banned_word in ["diagnose", "diagnosis", "treatment", "prescribe", "symptom", "suggest"]
            )
        )
        return filtered_answer if filtered_answer.strip() else "I'm here to provide information about X-ray analysis and pneumonia screening based on images."
    except Exception:
        return "Sorry, I'm currently unable to process your request. Please try again later."

def analyze_image(uploaded_file):
    with st.spinner("Analyzing your X-ray image..."):
        processed_image, original_image = process_image(uploaded_file)
        if processed_image is not None and pneumonia_model is not None:
            prediction = pneumonia_model.predict(processed_image)[0][0]
            # We do NOT show confidence or accuracy per instructions
            # Interpret prediction simply as binary classification without score
            if prediction > 0.5:
                result_class = "Signs possibly consistent with pneumonia detected."
                result_color = "result-pneumonia"
                explanation = (
                    "Analysis of the chest X-ray suggests patterns that may be associated with pneumonia, "
                    "such as areas of increased opacity or consolidation. "
                    "This is an automated screening tool, and results are for educational purposes only. "
                    "Please consult a healthcare professional for definitive diagnosis and advice."
                )
            else:
                result_class = "No clear signs of pneumonia detected."
                result_color = "result-normal"
                explanation = (
                    "The chest X-ray does not show evident patterns typically associated with pneumonia. "
                    "However, this tool does not replace professional medical evaluation. "
                    "Consult your healthcare provider for any concerns."
                )

            st.image(original_image, caption="Uploaded X-ray Image", use_container_width=True)

            st.markdown(f"""
            <div class="result-box {result_color}">
                <h3>{result_class}</h3>
                <p>{explanation}</p>
                <p class="disclaimer">
                    Medical Disclaimer: This tool provides educational screening information only and is not a substitute for professional medical advice, diagnosis, or treatment.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Log summary without confidence scores
            analysis_summary = result_class
            st.session_state.messages.append({
                "role": "assistant",
                "content": analysis_summary
            })
            return True
    return False

# Layout with two columns: left for upload and instructions, right for chat
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<h3>Upload Chest X-ray for Analysis</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Instructions:</strong>
        <ol>
            <li>Upload a chest X-ray image (PA or AP view) in JPG, JPEG, or PNG format.</li>
            <li>Wait for the analysis result to appear below the upload.</li>
            <li>Ask questions related to X-ray analysis or pneumonia screening in the chat box.</li>
        </ol>
        <p class="disclaimer">
            Note: This tool is an educational screening aid only and is not a medical diagnosis. Always seek professional healthcare advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload chest X-ray image", type=["jpg", "jpeg", "png"], key="file_uploader")

    if "last_processed_file" not in st.session_state:
        st.session_state.last_processed_file = None

    if uploaded_file is not None and uploaded_file != st.session_state.last_processed_file:
        st.session_state.last_processed_file = uploaded_file
        st.session_state.messages.append({"role": "user", "content": "I've uploaded a chest X-ray for analysis."})
        analyze_image(uploaded_file)

with col2:
    st.markdown("<h3>Healthcare Assistant Chat</h3>", unsafe_allow_html=True)
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask me about chest X-ray analysis or pneumonia screening...", key="user_input")
        submit_button = st.form_submit_button("Send")

        if submit_button and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = chat_response(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.experimental_rerun()

# Footer with disclaimer
st.markdown("""
<div class="footer">
    PneumoAssist is an educational tool designed for X-ray image screening support only.
    It does NOT provide medical diagnosis or treatment. Always consult qualified healthcare professionals for medical concerns.
</div>
""", unsafe_allow_html=True)
