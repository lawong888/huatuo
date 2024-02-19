import streamlit as st
import openai
import datetime
import os
import time
from glob import glob

# Assuming you have a similar configuration and utility functions as in your Gradio app.
import config
import speech_synthesis as ss
import db
import knowledge_bank as kb

# Hide footer in Streamlit (equivalent to your Gradio CSS)
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Set the character just like in your Gradio app.
character = "你係華陀, 一位生活喺東漢末年嘅中國醫師。現在你擔任PuraPharm的廣東品牌大使, 你會盡量用廣東話答問題唔可以用普通話回覆。"
openai.api_key = config.OPENAI_API_KEY

# Load images
image_files = glob(os.path.join("images", "*"))
image_dict = {os.path.basename(image_path).split(".")[0]: image_path for image_path in image_files}

# Display Header
st.markdown("# First Cantonese HyperChat Hua Tuo Chatbot Built by Humania.ai")

# Create columns for the video and text area
col1, col2 = st.columns([2, 1])

# Column 1 for promotional video and text area
with col1:
    st.video("https://www.purapharm.com/wp-content/uploads/2020/07/Nongs-video-Eng.mp4")
    st.text_area("From Farm to Bottle",
                 "農本方自設種子種苗繁育國家工程中心、生產及研究基地，以及連鎖中醫診所，將中藥的種植、生產製造及服用方法全面現代化，提供安全、可靠及有效的中藥產品及醫療服務。\n\n From our Chinese herbs plantation, to our state-of-the-art production facility, to Nong’s® Clinics, Nong’s® takes you to a journey to see how we modernize the way in which traditional Chinese medicine is manufactured, prepared and consumed, offering safe, reliable, and effective products and treatment for all.")
    
    # Image gallery
    st.image(image_dict.values(), caption=image_dict.keys(), width=100)

# Column 2 for the talking head video and audio input
with col2:
    st.video("huatuo.mp4")
    audio_input = st.file_uploader("Record your message:", type=['mp3', 'wav', 'ogg'])

# Functionality for transcribing, chatting, and synthesizing speech
def handle_audio(audio_input):
    if audio_input is not None:
        # Assuming you have a function similar to transcribe() in your Gradio app
        transcript_text = transcribe(audio_input)
        chat_response = chat(transcript_text)
        response_audio = speech_synthesis(chat_response)
        st.audio(response_audio)

# Submit button to handle the audio processing
if st.button("Submit"):
    handle_audio(audio_input)

# Clear button functionality
if st.button("Clear"):
    st.experimental_rerun()

# Function implementations would be required for:
# transcribe(audio) - To transcribe the audio input
# chat(transcript_text) - To get the chat response from the model
# speech_synthesis(chat_response) - To synthesize the response into audio

# As Streamlit does not support all Gradio functionalities out-of-the-box, such as real-time audio recording, 
# you would need to implement these or use a workaround such as file uploaders for audio input.
