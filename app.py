from flask import Flask, render_template, request
import sounddevice as sd
import numpy as np
import librosa
import joblib
from scipy.io.wavfile import write

app = Flask(__name__)

# Load model & encoder
# rf_model = joblib.load("random_forest_chord_model.pkl")
# label_encoder = joblib.load("label_encoder.pkl")

rf_model = joblib.load("random_forest_chord_model.pk1")
label_encoder = joblib.load("label_encoder.pk1")

# Feature extraction
def extract_chroma(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    y = librosa.util.normalize(y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma, axis=1)

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_chord = None

    if request.method == "POST":
        duration = 2        # seconds
        sample_rate = 22050

        # Record from mic
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1
        )
        sd.wait()

        # Save temp audio
        write("mic_input.wav", sample_rate, audio)

        # Extract features
        chroma = extract_chroma("mic_input.wav").reshape(1, -1)

        # Predict
        prediction = rf_model.predict(chroma)
        predicted_chord = label_encoder.inverse_transform(prediction)[0]

    return render_template("index.html", predicted_chord=predicted_chord)

if __name__ == "__main__":
    app.run(debug=True)


