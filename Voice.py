# To run the code type 'streamlit run Voice.py' in the terminal

import os
import streamlit as st
from openai import OpenAI
from pathlib import Path

st.title("Voice AI Chat")
st.write("Talk with AI using your voice. Built for the AI Generalist role at Crystal Group.")


api_key = st.text_input("Enter your OpenAI API Key", type="password")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a helpful voice assistant."}
    ]


if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()

    
    audio = st.audio_input("Speak now and press 🎙️ to speak again")

    if audio:
        # Function to convert the audio into text
        def transcribe(audio_blob):
            transcript = client.audio.transcriptions.create(
                file=audio_blob,
                model="whisper-1",
                response_format="text"
            )
            return transcript
        # Function get response from Openai
        def get_gpt_response(messages):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return response.choices[0].message.content

        def text_to_speech(text):
            speech_path = Path("response.mp3")
            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="nova",
                input=text
            ) as resp:
                resp.stream_to_file(speech_path)
            return speech_path

        
        user_text = transcribe(audio)
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        st.success(f"You said: {user_text}")

        
        ai_reply = get_gpt_response(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
        st.success(f"AI: {ai_reply}")

        
        speech_file = text_to_speech(ai_reply)
        with open(speech_file, "rb") as f:
            st.audio(f.read(), format="audio/mp3")

        st.divider()
        st.subheader("Chat History")
        for msg in st.session_state.chat_history[1:]:
            role = "You" if msg["role"] == "user" else "AI"
            st.markdown(f"**{role}:** {msg['content']}")

else:
    st.warning("Please enter your OpenAI API key to continue.")
