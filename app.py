# from flask import Flask, render_template, request
# import sounddevice as sd
# import numpy as np
# import librosa
# import joblib
# from scipy.io.wavfile import write
# from scipy.signal import butter, lfilter
# import noisereduce as nr

# app = Flask(__name__)

# # Load model & encoder
# # rf_model = joblib.load("random_forest_chord_model.pkl")
# # label_encoder = joblib.load("label_encoder.pkl")

# rf_model = joblib.load("random_forest_chord_model.pk1")
# label_encoder = joblib.load("label_encoder.pk1")

# def bandpass_filter(y, sr, lowcut=80, highcut=2000, order=5):
#     nyq = 0.5 * sr
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return lfilter(b, a, y)

# # Feature extraction
# # def extract_chroma(file_path):
# #     y, sr = librosa.load(file_path, sr=22050, mono=True)
# #     y, _ = librosa.effects.trim(y, top_db=20)
# #     y = librosa.util.normalize(y)
# #     chroma = librosa.feature.chroma_stft(y=y, sr=sr)
# #     return np.mean(chroma, axis=1)

# def extract_chroma(file_path):
#     y, sr = librosa.load(file_path, sr=22050, mono=True)

#     # 1️⃣ Remove unwanted frequencies
#     y = bandpass_filter(y, sr)

#     # 2️⃣ Reduce constant background noise
#     y = nr.reduce_noise(y=y, sr=sr)

#     # 3️⃣ Trim silence
#     y, _ = librosa.effects.trim(y, top_db=25)

#     # 4️⃣ Keep only harmonic part (chords)
#     y_harmonic, _ = librosa.effects.hpss(y)
#     y = librosa.util.normalize(y_harmonic)

#     # 5️⃣ Extract robust chroma features
#     chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

#     return np.mean(chroma, axis=1)



# @app.route("/", methods=["GET", "POST"])
# def index():
#     predicted_chord = None

#     if request.method == "POST":
#         duration = 2        # seconds
#         sample_rate = 22050

#         # Record from mic
#         audio = sd.rec(
#             int(duration * sample_rate),
#             samplerate=sample_rate,
#             channels=1
#         )
#         sd.wait()
        





#         # Save temp audio
#         write("mic_input.wav", sample_rate, audio)

#         # Extract features
#         chroma = extract_chroma("mic_input.wav").reshape(1, -1)

#         # Predict
#         prediction = rf_model.predict(chroma)
#         predicted_chord = label_encoder.inverse_transform(prediction)[0]

#     return render_template("index.html", predicted_chord=predicted_chord)

# if __name__ == "__main__":
#     app.run(debug=True)


# --new code here ->

from flask import Flask, render_template, request
import sounddevice as sd
import numpy as np
import librosa
import joblib
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter
import noisereduce as nr

app = Flask(__name__)

# =========================
# Load ML model & encoder
# =========================
rf_model = joblib.load("random_forest_chord_model.pk1")
label_encoder = joblib.load("label_encoder.pk1")


# =========================
# Band-pass filter
# =========================
def bandpass_filter(y, sr, lowcut=80, highcut=2000, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, y)


# =========================
# Feature extraction
# =========================
def extract_chroma(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)

    # 1️⃣ Remove unwanted frequencies
    y = bandpass_filter(y, sr)

    # 2️⃣ Reduce background noise
    y = nr.reduce_noise(y=y, sr=sr)

    # 3️⃣ Trim silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # 4️⃣ Keep harmonic content only
    y_harmonic, _ = librosa.effects.hpss(y)
    y = librosa.util.normalize(y_harmonic)

    # 5️⃣ Extract chroma features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    return np.mean(chroma, axis=1)


# =========================
# Flask route
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    predicted_chord = None

    if request.method == "POST":
        duration = 2        # seconds
        sample_rate = 22050

        # 🎤 Record from microphone
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1
        )
        sd.wait()

        # 🔇 Silence / noise check (IMPORTANT)
        energy = np.mean(audio ** 2)
        if energy < 0.001:
            predicted_chord = "Too much noise or too soft — play louder"
            return render_template(
                "index.html",
                predicted_chord=predicted_chord
            )

        # 💾 Save temporary audio
        write("mic_input.wav", sample_rate, audio)

        # 🎼 Extract features
        chroma = extract_chroma("mic_input.wav").reshape(1, -1)

        # 🤖 Predict chord
        prediction = rf_model.predict(chroma)
        predicted_chord = label_encoder.inverse_transform(prediction)[0]

    return render_template("index.html", predicted_chord=predicted_chord)


# =========================
# Run app
# =========================
if __name__ == "__main__":
    app.run(debug=True)


## new code works fine