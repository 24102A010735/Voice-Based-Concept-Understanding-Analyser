import streamlit as st
from src.speech_to_text import SpeechToText
from src.semantic_analysis import SemanticAnalyzer

st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Voice-Based Concept Understanding Analyser")

st.write("Upload your voice explanation.")

topic = st.selectbox(
    "Select Topic",
    [
        "Machine Learning",
        "Artificial Intelligence",
        "Cloud Computing",
        "Data Science"
    ]
)

audio = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3", "m4a"]
)

if audio:

    st.audio(audio)

    if st.button("Analyze"):

        with st.spinner("Transcribing..."):

            stt = SpeechToText()
            transcript = stt.transcribe_audio(audio)

        semantic = SemanticAnalyzer()
        score, feedback = semantic.analyze(topic, transcript)

        st.success("Transcription Complete!")

        st.subheader("📝 Transcript")
        st.write(transcript)

        st.subheader("🧠 Understanding Score")

        st.metric(
            label="Similarity Score",
            value=f"{score}%"
        )

        st.success(feedback)