from dotenv import load_dotenv, find_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from google import genai
from google.genai import types
import wave
import os
import streamlit as st

load_dotenv(find_dotenv())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Make sure to set this in your .env
WAV_FILE_NAME = 'out.wav'
IMG_FILE_NAME = "image.jpg"

client = genai.Client(api_key=GEMINI_API_KEY)

#imgtotext to create a scenario form the picture
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]["generated_text"]
    return text

##llm to generate a short story using gpt api key
def generate_story(scenario):
    prompt = f"""
    You are a story teller;
    You can generate a short story based on a simple narrative, the story should be no more than 50 words;
    
    CONTEXT: {scenario}
    STORY:
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    story = response.text
    print(story)
    return story

# Set up the wave file to save the output:
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
   with wave.open(filename, "wb") as wf:
      wf.setnchannels(channels)
      wf.setsampwidth(sample_width)
      wf.setframerate(rate)
      wf.writeframes(pcm)

# text to speech
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
    wave_file(WAV_FILE_NAME, data) # Saves the file to current directory

def main():
    st.set_page_config(
        page_title="ImagiNarrate AI", 
        page_icon="🎙️",
        layout="wide"
    )
    st.header("🎨 Transform Images into Spoken Stories")
    st.caption("Upload any image and let AI create and narrate its story")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(IMG_FILE_NAME, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        scenario = img2text(IMG_FILE_NAME)
        story = generate_story(scenario)
        text2speech(story)
        
        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        
        st.audio(WAV_FILE_NAME)

if __name__ == '__main__':
    main()