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

# Load environment variables from Streamlit secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("❌ GEMINI_API_KEY not found in Streamlit secrets. Please configure it in the Secrets tab.")
    st.stop()

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
    try:
        with open("usage_tracker.json", "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        init_usage_tracking()
        return True, 0
    
    # Check if we need to reset the counter (new hour)
    last_reset = datetime.fromisoformat(data["last_reset"])
    now = datetime.now()
    
    if now - last_reset >= timedelta(hours=1):
        # Reset the counter
        data["usage_count"] = 0
        data["last_reset"] = now.isoformat()
        data["usage_history"] = []
        
        with open("usage_tracker.json", "w") as f:
            json.dump(data, f)
    
    # Check if limit exceeded
    if data["usage_count"] >= 3:
        return False, data["usage_count"]
    
    return True, data["usage_count"]

def increment_usage():
    """Increment the usage counter"""
    try:
        with open("usage_tracker.json", "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        init_usage_tracking()
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
        model="gemini-1.5-flash",
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
        model="gemini-2.5-flash-preview-tts",
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

def render_santostar_promotion():
    """Render Santo Star Limited promotional elements"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Powered by Santo Star Limited")
    st.sidebar.markdown("""
    **Enterprise AI Solutions:**
    - Custom AI Storytelling
    - Unlimited Usage Plans
    - White-label Solutions
    - API Integration
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📧 Contact for Full Access")
    st.sidebar.markdown("""
    **Email:** inquiries@santostar.com  
    **Website:** www.santostar.com  
    **Plans:** Starter, Business, Enterprise
    """)
    
    if st.sidebar.button("📩 Request Enterprise Demo"):
        st.sidebar.success("Contact inquiries@santostar.com for a custom demo!")

def render_rate_limit_message(current_usage, is_allowed):
    """Render rate limit information with Santo Star promotion"""
    if not is_allowed:
        st.error("""
        ❌ **Rate Limit Exceeded!**
        
        This demo allows **3 story generations per hour** across all users.
        
        🚀 **For unlimited access and enterprise features:**
        - **Custom AI solutions**
        - **Higher rate limits**
        - **White-label applications**
        - **API integration**
        
        📧 **Contact:** inquiries@santostar.com
        """)
        
        try:
            with open("usage_tracker.json", "r") as f:
                data = json.load(f)
            last_reset_time = datetime.fromisoformat(data["last_reset"])
            next_reset_time = last_reset_time + timedelta(hours=1)
            st.info(f"⏰ **Next reset:** {next_reset_time.strftime('%H:%M:%S')}")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            st.info("⏰ **Next reset:** In 1 hour")
        
        return False
    else:
        if current_usage >= 2:  # Warning when approaching limit
            st.warning(f"⚠️ **Usage Alert:** {current_usage}/3 generations used this hour. Contact inquiries@santostar.com for unlimited access.")
        return True

def main():
    st.set_page_config(
        page_title="ImagiNarrate AI by Santo Star", 
        page_icon="🎙️",
        layout="wide"
    )
    
    # Initialize usage tracking
    init_usage_tracking()
    
    # Main header with Santo Star branding
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("🎨 ImagiNarrate AI - Transform Images into Spoken Stories")
        st.caption("Powered by Santo Star Limited • Upload any image and let AI create and narrate its story")
    with col2:
        st.markdown("""
        <div style='text-align: right; padding: 10px; background: #f0f2f6; border-radius: 5px;'>
            <strong>Santo Star Limited</strong><br>
            <small>AI Innovation</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Display usage information
    is_allowed, current_usage = check_rate_limit()
    
    # Sidebar content
    with st.sidebar:
        st.title("📊 Usage Dashboard")
        st.metric("Generations This Hour", f"{current_usage}/3")
        
        if not is_allowed:
            st.error("❌ Limit Exceeded")
        elif current_usage >= 2:
            st.warning("⚠️ Approaching Limit")
        else:
            st.success("✅ Within Limit")
        
        # Santo Star promotion
        render_santostar_promotion()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Use temporary file for better file handling
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
        
        if st.button("✨ Generate Story & Audio", type="primary"):
            # Check rate limit before processing
            is_allowed, current_usage = check_rate_limit()
            if not render_rate_limit_message(current_usage, is_allowed):
                # Clean up temporary file
                os.unlink(tmp_file_path)
                return
            
            with st.spinner("🔍 Analyzing image..."):
                scenario = img2text(tmp_file_path)
            
            with st.spinner("📖 Generating story..."):
                story = generate_story(scenario)
            
            with st.spinner("🔊 Converting to speech..."):
                text2speech(story)
            
            # Increment usage counter after successful generation
            new_usage_count = increment_usage()
            
            # Update sidebar metric
            st.sidebar.metric("Generations This Hour", f"{new_usage_count}/3")
            
            # Success message with promotion
            st.success("🎉 Story generated successfully!")
            st.info("💡 **Want unlimited access?** Contact inquiries@santostar.com for enterprise plans!")
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("📋 Scenario from Image"):
                    st.write(scenario)
            
            with col2:
                with st.expander("📖 Generated Story"):
                    st.write(story)
            
            # Audio player
            st.subheader("🎧 Listen to Your Story")
            st.audio(WAV_FILE_NAME)
            
            # Download button for audio
            with open(WAV_FILE_NAME, "rb") as audio_file:
                st.download_button(
                    label="⬇️ Download Audio",
                    data=audio_file,
                    file_name="santostar_ai_story.wav",
                    mime="audio/wav"
                )
            
            # Enterprise CTA section
            st.markdown("---")
            st.markdown("### 🚀 Ready for More?")
            st.markdown("""
            **Santo Star Limited offers enterprise solutions:**
            - ✅ **Unlimited story generations**
            - ✅ **Custom AI models**
            - ✅ **Batch processing**
            - ✅ **API access**
            - ✅ **White-label applications**
            
            **Contact us today:** inquiries@santostar.com
            """)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    # Footer with Santo Star branding
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <strong>ImagiNarrate AI</strong> • Powered by <strong>Santo Star Limited</strong> • 
        <a href="mailto:inquiries@santostar.com" style='color: #007bff;'>inquiries@santostar.com</a> • 
        <a href="https://www.santostar.com" style='color: #007bff;'>www.santostar.com</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()