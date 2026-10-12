# 🎧 Music Genre Classification

A machine learning project that predicts the genre of a song from its audio.
Upload a track in the web app and the model returns one of 10 genres.

## Features
- Extracts 29 audio features per song with Librosa
- Compares two classical ML models: SVM and Random Forest
- Streamlit web app with Home, Classify, Dashboard and About pages
- Dashboard with accuracy, F1-score per genre and a confusion matrix

## How it works
```
Audio file → Librosa (first 30 s) → 29 features → StandardScaler → SVM → Genre
```

**Features (29 total):** 13 MFCC, 12 chroma, spectral centroid, spectral bandwidth, spectral rolloff, zero crossing rate (each averaged over time).

## Dataset
[GTZAN Music Genre Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification):
about 1,000 tracks, 10 genres (blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock).
One file in the jazz folder is corrupted, so 999 songs are used.
The dataset is not included in this repo because of its size.

## Results
Stratified 80/20 train-test split, `random_state=42`.

| Model | Test accuracy |
|---|---|
| SVM (RBF) | 60.5% |
| Random Forest | 57.0% |

Classical, metal and jazz are classified best. Country, disco and hip-hop are the hardest, since they overlap with other genres.

## Project structure
```
music-genre-classifier/
├── app.py                 # Streamlit app
├── src/
│   ├── extract_features.py
│   ├── train_model.py
│   └── predict.py
├── models/                # saved model and scaler
├── features.csv
└── requirements.txt
```

## Setup
```bash
git clone https://github.com/YOUR-USERNAME/music-genre-classifier.git
cd music-genre-classifier
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run app.py
```

## Retrain from scratch (optional)
1. Download GTZAN and place the genre folders in `dataset/genres/`
2. `python src/extract_features.py`
3. `python src/train_model.py`

## Limitations
- GTZAN contains older recordings, so modern songs can be misclassified.
- Features are averages, so lyrics and song structure are ignored.
- Percentages in the app are model scores, not calibrated probabilities.

## Tech stack
Python, Librosa, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Joblib, Streamlit
