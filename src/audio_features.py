import librosa
import numpy as np


class AudioFeatures:

    def analyze(self, audio_path):

        y, sr = librosa.load(audio_path, sr=None)

        duration = librosa.get_duration(y=y, sr=sr)

        rms = librosa.feature.rms(y=y)[0]

        average_energy = float(np.mean(rms))

        return {
            "duration": round(duration, 2),
            "energy": round(average_energy, 4)
        }