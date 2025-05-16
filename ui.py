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

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f9fbff;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333333;
    }
    .info-box {
        background-color: #e7f0ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        font-size: 14px;
        color: #1f4e79;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 2px 6px rgb(0 0 0 / 0.1);
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
        color: #555;
        font-style: italic;
        margin-top: 10px;
    }
    .chat-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        height: 400px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #e1efff;
        padding: 10px;
        border-radius: 15px 15px 0 15px;
        margin: 10px 0;
        max-width: 80%;
        float: right;
        clear: both;
        color: #1a3e72;
    }
    .assistant-message {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 15px 15px 15px 0;
        margin: 10px 0;
        max-width: 80%;
        float: left;
        clear: both;
        color: #2f3e4d;
    }
    .header-section {
        text-align: center;
        margin-bottom: 20px;
        font-weight: 700;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Application header
st.markdown("<div class='header-section'><h1>🫁 PneumoAssist: X-Ray Analysis & Healthcare Assistant</h1></div>", unsafe_allow_html=True)

# Load pneumonia model with caching for efficiency
@st.cache_resource
def load_pneumonia_model():
    try:
        model = load_model('cnn_model91.h5')
        return model
    except Exception as e:
        st.error(f"Error loading pneumonia model: {str(e)}")
        return None

# Load model
pneumonia_model = load_pneumonia_model()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial greeting message
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hello! I'm PneumoAssist, your medical assistant for pneumonia detection and information. "
            "I can analyze chest X-rays and answer your questions about pneumonia. How can I help you today?"
        )
    })

def process_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        display_image = image.copy()
        image = image.convert('L')  # Grayscale
        image = image.resize((150, 150))  # Model input size
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Batch dim
        return image_array, display_image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def analyze_image(uploaded_file):
    with st.spinner("Analyzing your X-ray image..."):
        processed_image, original_image = process_image(uploaded_file)
        if processed_image is not None and pneumonia_model is not None:
            prediction = pneumonia_model.predict(processed_image)[0][0]
            # We DO NOT display the confidence percentages or scores per your request
            if prediction > 0.5:
                result_class = "Signs possibly consistent with pneumonia detected."
                result_color = "result-pneumonia"
                explanation = (
                    "Analysis suggests patterns that may be associated with pneumonia, such as increased opacity or consolidation. "
                    "This is an automated educational screening tool only. "
                    "Please consult a healthcare professional for diagnosis and treatment."
                )
            else:
                result_class = "No clear signs of pneumonia detected."
                result_color = "result-normal"
                explanation = (
                    "The X-ray does not show patterns typically associated with pneumonia. "
                    "This tool does not replace professional medical evaluation. "
                    "Consult your healthcare provider if you have any concerns."
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

            st.session_state.messages.append({
                "role": "assistant",
                "content": result_class
            })
            return True
    return False

def chat_response(user_input):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

    # Use your original chat system with no changes here

    history = st.session_state.messages[-5:-1] if len(st.session_state.messages) > 1 else []
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

    prompt = f"""<s>[INST] <<SYS>>
You are PneumoAssist, a concise assistant. You are also able to predict pneumonia based on x ray images with a 91% accuracy when the user submits an image. 
Respond briefly and shortly to the user's input, considering the recent conversation. Do NOT continue conversations or assume symptoms or anything beyond the immediate query.
<</SYS>>

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

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    full_text = response.json()[0]['generated_text']
    return full_text.split("[/INST]")[-1].strip()

# Layout columns
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<h3>Upload X-ray for Analysis</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>How to use:</strong>
        <ol>
            <li>Upload a chest X-ray image (PA or AP view)</li>
            <li>Wait for the analysis results</li>
            <li>Ask follow-up questions in the chat</li>
        </ol>
        <p class="disclaimer">
            This tool analyzes chest X-rays for potential signs of pneumonia. Always consult healthcare professionals for diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"], key="file_uploader")

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

    # Your original chat form and logic unchanged
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask me about pneumonia or the X-ray analysis...", key="user_input")
        submit_button = st.form_submit_button("Send")
        if submit_button and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = chat_response(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.experimental_rerun()

# Footer with disclaimers and credits
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
    <p class="disclaimer">PneumoAssist is an educational tool and should not replace professional medical advice.
    The X-ray analysis is performed using a convolutional neural network trained on chest X-ray datasets.
    Always consult with qualified healthcare professionals for proper diagnosis and treatment.</p>
    <p style="font-size:12px; margin-top:10px;">
        Created by Karim Derbali, Terry Zhuang, Yunlei Xu, Muhammad Hassaan Sohail,<br>
        &copy; University of Chicago.<br>
        Your privacy and data security are important to us. Uploaded images are processed only in-memory and not stored or shared.
    </p>
</div>
""", unsafe_allow_html=True)
