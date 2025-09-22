from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from google import genai
from google.genai import types
import wave
import os
import streamlit as st
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta

# Load environment variables
load_dotenv(find_dotenv())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WAV_FILE_NAME = 'out.wav'

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Rate limiting functions
def init_usage_tracking():
    """Initialize the usage tracking file if it doesn't exist"""
    if not Path("usage_tracker.json").exists():
        with open("usage_tracker.json", "w") as f:
            json.dump({"usage_count": 0, "last_reset": datetime.now().isoformat(), "usage_history": []}, f)

def check_rate_limit():
    """Check if the rate limit has been exceeded (max 3 uses per hour)"""
    with open("usage_tracker.json", "r") as f:
        data = json.load(f)
    
    # Check if we need to reset the counter (new hour)
    last_reset = datetime.fromisoformat(data["last_reset"])
    now = datetime.now()
    
    if now - last_reset >= timedelta(hours=1):
        # Reset the counter
        data["usage_count"] = 0
        data["last_reset"] = now.isoformat()
        data["usage_history"] = []
    
    # Check if limit exceeded
    if data["usage_count"] >= 3:
        return False, data["usage_count"]
    
    return True, data["usage_count"]

def increment_usage():
    """Increment the usage counter"""
    with open("usage_tracker.json", "r") as f:
        data = json.load(f)
    
    # Check if we need to reset the counter (new hour)
    last_reset = datetime.fromisoformat(data["last_reset"])
    now = datetime.now()
    
    if now - last_reset >= timedelta(hours=1):
        # Reset the counter
        data["usage_count"] = 0
        data["last_reset"] = now.isoformat()
        data["usage_history"] = []
    
    # Increment usage
    data["usage_count"] += 1
    data["usage_history"].append({
        "timestamp": now.isoformat(),
        "action": "generate_story"
    })
    
    with open("usage_tracker.json", "w") as f:
        json.dump(data, f)
    
    return data["usage_count"]

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
    
    # Initialize usage tracking
    init_usage_tracking()
    
    st.header("🎨 Transform Images into Spoken Stories")
    st.caption("Upload any image and let AI create and narrate its story")
    
    # Display usage information
    is_allowed, current_usage = check_rate_limit()
    st.sidebar.info(f"**Usage this hour:** {current_usage}/3")
    
    if not is_allowed:
        st.sidebar.error("❌ Rate limit exceeded")
        last_reset_time = datetime.fromisoformat(json.load(open("usage_tracker.json"))["last_reset"])
        next_reset_time = last_reset_time + timedelta(hours=1)
        st.sidebar.write(f"Next reset: {next_reset_time.strftime('%H:%M:%S')}")
    
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
            # Check rate limit before processing
            is_allowed, current_usage = check_rate_limit()
            if not is_allowed:
                st.error("""
                ❌ Rate limit exceeded!
                
                This demo allows only 3 story generations per hour across all users.
                Please try again later or contact us for increased access.
                """)
                return
            
            with st.spinner("🔍 Analyzing image..."):
                scenario = img2text(tmp_file_path)
            
            with st.spinner("📖 Generating story..."):
                story = generate_story(scenario)
            
            with st.spinner("🔊 Converting to speech..."):
                text2speech(story)
            
            # Increment usage counter after successful generation
            new_usage_count = increment_usage()
            st.sidebar.info(f"**Usage this hour:** {new_usage_count}/3")
            
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