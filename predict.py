import sys
import joblib
import librosa
import numpy as np
import pandas as pd

MODEL_PATH = "models/genre_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Same column names and order as features.csv (used during training)
FEATURE_NAMES = (
    [f"mfcc_{i+1}" for i in range(13)]
    + [f"chroma_{i+1}" for i in range(12)]
    + ["spectral_centroid", "spectral_bandwidth",
       "spectral_rolloff", "zero_crossing_rate"]
)


def extract_features(file_path):
    """Convert one audio file into 29 numbers (same as extract_features.py)."""
    y, sr = librosa.load(file_path, duration=30)

    features = []

    # 13 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))

    # 12 Chroma values
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    # Spectral centroid, bandwidth, rolloff, zero crossing rate
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))

    # One row (one song) with the same column names used in training
    return pd.DataFrame([features], columns=FEATURE_NAMES)


def predict_genre(file_path):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features = extract_features(file_path)
    scaled_features = scaler.transform(features)  # same scaler as training
    prediction = model.predict(scaled_features)

    return prediction[0]


def predict_with_scores(file_path):
    """Return (predicted genre, [(genre, score), ...] sorted high to low).

    The scores are model scores, not guaranteed true probabilities,
    because the SVM is not calibrated.
    """
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features = extract_features(file_path)
    scaled_features = scaler.transform(features)

    genre = model.predict(scaled_features)[0]
    scores = model.predict_proba(scaled_features)[0]
    ranked = sorted(zip(model.classes_, scores), key=lambda x: x[1], reverse=True)

    return genre, ranked


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/song.wav")
        sys.exit()

    genre = predict_genre(sys.argv[1])
    print("\nPredicted Genre:")
    print(genre.upper())