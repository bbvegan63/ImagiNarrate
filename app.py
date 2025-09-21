from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from google import genai
from google.genai import types
import wave
import os
import streamlit as st
import tempfile

load_dotenv(find_dotenv())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WAV_FILE_NAME = 'out.wav'

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Image to text function
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]["generated_text"]
    return text

# Generate story using Gemini
def generate_story(scenario):
    prompt = f"""
    You are a story teller;
    You can generate a short story based on a simple narrative, the story should be no more than 50 words;
    
    CONTEXT: {scenario}
    STORY:
    """
    
    response = client.models.generate_content(
        model="gemini-1.5-flash",  # Using a more stable model
        contents=prompt
    )
    
    story = response.text
    return story

# Save wave file function
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

# Text to speech function
def text2speech(message):
    template = f"Say cheerfully: {message}"

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",  # Updated model name
        contents=template,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name='Kore',
                    )
                )
            ),
        )
    )

    data = response.candidates[0].content.parts[0].inline_data.data
    wave_file(WAV_FILE_NAME, data)

def main():
    st.set_page_config(
        page_title="ImagiNarrate AI", 
        page_icon="🎙️",
        layout="wide"
    )
    st.header("🎨 Transform Images into Spoken Stories")
    st.caption("Upload any image and let AI create and narrate its story")
    
    # Check if API key is configured
    if not GEMINI_API_KEY:
        st.error("❌ Please set the GEMINI_API_KEY in your .env file")
        return
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Use temporary file for better file handling
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Fixed: use_container_width instead of use_column_width
        st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
        
        if st.button("✨ Generate Story & Audio", type="primary"):
            with st.spinner("🔍 Analyzing image..."):
                scenario = img2text(tmp_file_path)
            
            with st.spinner("📖 Generating story..."):
                story = generate_story(scenario)
            
            with st.spinner("🔊 Converting to speech..."):
                text2speech(story)
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("📋 Scenario from Image"):
                    st.write(scenario)
            
            with col2:
                with st.expander("📖 Generated Story"):
                    st.write(story)
            
            # Audio player
            st.audio(WAV_FILE_NAME)
            
            # Download button for audio
            with open(WAV_FILE_NAME, "rb") as audio_file:
                st.download_button(
                    label="⬇️ Download Audio",
                    data=audio_file,
                    file_name="ai_generated_story.wav",
                    mime="audio/wav"
                )
            
            # Clean up temporary file
            os.unlink(tmp_file_path)

if __name__ == '__main__':
    main()