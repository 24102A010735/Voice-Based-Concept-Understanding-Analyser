import whisper
import tempfile
import os


class SpeechToText:

    def __init__(self):
        self.model = whisper.load_model("base")

    def transcribe_audio(self, uploaded_file):

        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        result = self.model.transcribe(temp_path)

        os.remove(temp_path)

        return result["text"]