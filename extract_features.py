import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "dataset/genres"
OUTPUT_FILE = "features.csv"


def extract_features(file_path):
    """Convert one audio file into a list of 29 numbers."""
    try:
        # Load the first 30 seconds of the song
        y, sr = librosa.load(file_path, duration=30)

        features = []

        # 13 MFCCs (average over time)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))

        # 12 Chroma values (average over time)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))

        # Spectral centroid, bandwidth, rolloff, zero crossing rate
        features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.zero_crossing_rate(y)))

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    data = []
    print("Starting feature extraction...")

    for genre in sorted(os.listdir(DATASET_PATH)):
        genre_path = os.path.join(DATASET_PATH, genre)
        if not os.path.isdir(genre_path):
            continue

        print(f"\nProcessing genre: {genre}")

        for filename in sorted(os.listdir(genre_path)):
            if not filename.lower().endswith(".wav"):
                continue

            features = extract_features(os.path.join(genre_path, filename))

            if features is not None:
                features.append(genre)  # folder name = genre label
                data.append(features)
                print(f"Processed: {filename}")

    columns = (
        [f"mfcc_{i+1}" for i in range(13)]
        + [f"chroma_{i+1}" for i in range(12)]
        + ["spectral_centroid", "spectral_bandwidth",
           "spectral_rolloff", "zero_crossing_rate", "genre"]
    )

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nFeature extraction completed.")
    print(f"Total samples: {len(df)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print(df["genre"].value_counts())


if __name__ == "__main__":
    main()