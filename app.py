import streamlit as st
from src.speech_to_text import SpeechToText
from src.semantic_analysis import SemanticAnalyzer
from src.audio_features import AudioFeatures

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

            result = stt.transcribe_audio(audio)

            st.write(result)

            st.stop()

        st.write("✅ Transcript completed")

        semantic = SemanticAnalyzer()

        st.write("✅ Model loaded")

        score, feedback = semantic.analyze(topic, transcript)

        st.write("✅ Analysis completed")

        audio_analysis = AudioFeatures()

        features = audio_analysis.analyze(audio_path)

        st.success("Transcription Complete!")

        st.subheader("📝 Transcript")
        st.write(transcript)

        st.subheader("🧠 Understanding Score")

        st.metric(
            "Similarity Score",
            f"{score}%"
        )

        st.success(feedback)

        st.subheader("🎙 Audio Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Duration",
                f"{features['duration']} sec"
            )

        with col2:
            st.metric(
                "Voice Energy",
                features["energy"]
            )