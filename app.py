import os
import tempfile

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.predict import MODEL_PATH, SCALER_PATH, predict_with_scores

st.set_page_config(page_title="Genre Classifier", page_icon="🎧", layout="wide")

# Display name and pastel colour for each genre
GENRES = {
    "blues": ("Blues", "#9DB4C0"),
    "classical": ("Classical", "#C9B79C"),
    "country": ("Country", "#D4A373"),
    "disco": ("Disco", "#D8A7B1"),
    "hiphop": ("Hip-Hop", "#B8A1C9"),
    "jazz": ("Jazz", "#E0C58F"),
    "metal": ("Metal", "#8E8A93"),
    "pop": ("Pop", "#EFB3C0"),
    "reggae": ("Reggae", "#A3B18A"),
    "rock": ("Rock", "#C98B7B"),
}

BG = "#F6F1EB"
SAGE = "#8A9A7B"
INK = "#3A3330"
MUTED = "#8B8178"

# ======================================================================
# Styling
# ======================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600&display=swap');

html, body, .stApp { font-family: 'Inter', 'Segoe UI', sans-serif; }
.stApp { background: #F6F1EB; }
.block-container { max-width: 1050px; padding-top: 2.5rem; padding-bottom: 4rem; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

h1, h2, h3, .serif { font-family: 'DM Serif Display', Georgia, serif !important; color: #3A3330; font-weight: 400; }

/* Sidebar */
[data-testid="stSidebar"] { background: #EFE7DD; border-right: 1px solid #E3D9CC; }
.brand { font-family: 'DM Serif Display', Georgia, serif; font-size: 1.5rem; color: #3A3330; margin: 0.5rem 0 0.1rem 0; }
.brand-sub { font-size: 0.75rem; letter-spacing: 2px; color: #8B8178; margin-bottom: 1.5rem; }

/* Hero */
.eyebrow { font-size: 0.75rem; letter-spacing: 3px; text-transform: uppercase; color: #8A9A7B; font-weight: 600; margin-bottom: 0.8rem; }
.hero-title { font-family: 'DM Serif Display', Georgia, serif; font-size: 4.2rem; line-height: 1.02; color: #3A3330; margin: 0 0 1rem 0; }
.hero-title em { color: #C98B7B; font-style: italic; }
.hero-sub { color: #8B8178; font-size: 1.1rem; max-width: 560px; line-height: 1.6; margin-bottom: 1.5rem; }

/* Cards */
.card { background: #FFFDF9; border: 1px solid #ECE3D7; border-radius: 22px; padding: 1.4rem 1.5rem; height: 100%; box-shadow: 0 6px 24px rgba(90, 70, 50, 0.05); }
.card .num { font-family: 'DM Serif Display', Georgia, serif; font-size: 2.4rem; color: #3A3330; line-height: 1.1; }
.card .lbl { font-size: 0.8rem; letter-spacing: 1.5px; text-transform: uppercase; color: #8B8178; margin-top: 0.3rem; }
.card h4 { font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size: 1.25rem; color: #3A3330; margin: 0.5rem 0 0.3rem 0; }
.card p { color: #8B8178; font-size: 0.92rem; line-height: 1.55; margin: 0; }
.step { display: inline-block; width: 32px; height: 32px; line-height: 32px; text-align: center; border-radius: 50%; background: #E4EAD9; color: #6B7C5C; font-weight: 600; font-size: 0.9rem; }

.section-title { font-family: 'DM Serif Display', Georgia, serif; font-size: 1.8rem; color: #3A3330; margin: 2.5rem 0 1rem 0; }

/* Uploader */
[data-testid="stFileUploaderDropzone"] { background: #FFFDF9; border: 1.5px dashed #CFC3B3; border-radius: 22px; padding: 2rem; }

/* Buttons */
.stButton > button { border: none; border-radius: 999px; padding: 0.65rem 1.8rem; font-weight: 600; background: #8A9A7B; color: white; transition: all .2s ease; }
.stButton > button:hover { background: #76866A; color: white; transform: translateY(-1px); box-shadow: 0 8px 20px rgba(138, 154, 123, 0.35); }

/* Result */
.result { text-align: center; border-radius: 28px; padding: 2.5rem 1.5rem; margin-top: 1.2rem; border: 1px solid #ECE3D7; }
.result .lbl { font-size: 0.75rem; letter-spacing: 4px; color: #3A3330aa; }
.result .name { font-family: 'DM Serif Display', Georgia, serif; font-size: 4rem; color: #3A3330; line-height: 1.1; margin-top: 0.4rem; }

/* Score bars */
.bar-row { display: flex; align-items: center; gap: 12px; margin: 0.7rem 0; }
.bar-name { width: 90px; font-size: 0.9rem; color: #3A3330; }
.bar-track { flex: 1; height: 10px; background: #EFE7DD; border-radius: 999px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; }
.bar-val { width: 48px; text-align: right; font-size: 0.85rem; color: #8B8178; }

/* Pills */
.pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.8rem; }
.pill { padding: 6px 16px; border-radius: 999px; font-size: 0.85rem; border: 1px solid #E3D9CC; color: #8B8178; background: #FFFDF9; }
.pill.on { color: #3A3330; font-weight: 600; border-color: transparent; }

.note { color: #8B8178; font-size: 0.82rem; margin-top: 0.8rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def render(html):
    """Show HTML. Indentation and blank lines are stripped so markdown
    doesn't treat the HTML as a code block."""
    cleaned = "\n".join(line.strip() for line in html.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# ======================================================================
# Evaluation (recomputes the SAME test split used in train_model.py)
# ======================================================================
@st.cache_data
def load_evaluation():
    df = pd.read_csv("features.csv")
    X = df.drop("genre", axis=1)
    y = df["genre"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # SVM was trained on scaled features, Random Forest on original ones
    if isinstance(model, RandomForestClassifier):
        X_eval = X_test
        model_name = "Random Forest"
    else:
        X_eval = scaler.transform(X_test)
        model_name = "SVM"

    preds = model.predict(X_eval)
    classes = sorted(y.unique())

    return {
        "model_name": model_name,
        "n_songs": len(df),
        "n_genres": len(classes),
        "n_features": X.shape[1],
        "n_test": len(y_test),
        "accuracy": accuracy_score(y_test, preds),
        "report": classification_report(y_test, preds, output_dict=True),
        "cm": confusion_matrix(y_test, preds, labels=classes),
        "classes": classes,
        "counts": y.value_counts().to_dict(),
    }


def get_evaluation():
    try:
        return load_evaluation()
    except Exception:
        return None


def style_axes(ax):
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)


# ======================================================================
# Sidebar navigation
# ======================================================================
st.session_state.setdefault("page", "Home")

with st.sidebar:
    render(
        """
        <div class="brand">Genre Classifier</div>
        <div class="brand-sub">MACHINE LEARNING PROJECT</div>
        """
    )
    page = st.radio(
        "Navigate",
        ["Home", "Classify", "Dashboard", "About"],
        key="page",
        label_visibility="collapsed",
    )


def go_to_classify():
    st.session_state["page"] = "Classify"


# ======================================================================
# Page: Home
# ======================================================================
def page_home():
    ev = get_evaluation()

    render(
        """
        <div class="eyebrow">Audio · Machine Learning · Python</div>
        <div class="hero-title">Hear it.<br><em>Name the genre.</em></div>
        <div class="hero-sub">Upload any song and the model measures 29 audio features,
        then predicts which of 10 genres it belongs to.</div>
        """
    )
    st.button("Try the classifier", on_click=go_to_classify)

    st.write("")
    cols = st.columns(4)
    if ev:
        stats = [
            (f"{ev['n_songs']}", "Songs in dataset"),
            (f"{ev['n_genres']}", "Genres"),
            (f"{ev['n_features']}", "Audio features"),
            (f"{ev['accuracy'] * 100:.1f}%", "Test accuracy"),
        ]
    else:
        stats = [("999", "Songs"), ("10", "Genres"), ("29", "Features"), ("-", "Accuracy")]

    for col, (num, lbl) in zip(cols, stats):
        with col:
            render(f'<div class="card"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>')

    render('<div class="section-title">How it works</div>')
    steps = [
        ("1", "Upload", "Drop in a WAV, MP3, OGG or FLAC file."),
        ("2", "Extract", "Librosa turns the first 30 seconds into 29 numbers."),
        ("3", "Scale", "StandardScaler puts every feature on the same range."),
        ("4", "Predict", "A trained SVM picks the most likely genre."),
    ]
    cols = st.columns(4)
    for col, (n, title, text) in zip(cols, steps):
        with col:
            render(f'<div class="card"><span class="step">{n}</span><h4>{title}</h4><p>{text}</p></div>')

    render('<div class="section-title">Genres it knows</div>')
    pills = "".join(
        f'<span class="pill on" style="background:{c}55;">{n}</span>'
        for n, c in GENRES.values()
    )
    render(f'<div class="pills">{pills}</div>')


# ======================================================================
# Page: Classify
# ======================================================================
def page_classify():
    render(
        """
        <div class="eyebrow">Classifier</div>
        <h1 style="margin:0 0 0.4rem 0;">Upload a track</h1>
        <div class="hero-sub" style="margin-bottom:1rem;">WAV, MP3, OGG or FLAC.
        The first 30 seconds are analysed.</div>
        """
    )

    uploaded_file = st.file_uploader(
        "Upload a song", type=["wav", "mp3", "ogg", "flac"], label_visibility="collapsed"
    )

    if uploaded_file is None:
        return

    st.audio(uploaded_file)

    if st.button("Analyse track"):
        with st.spinner("Listening..."):
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            try:
                genre, ranked = predict_with_scores(temp_path)
            except Exception as e:
                st.error(f"Error: {e}")
                return
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        name, color = GENRES.get(genre, (genre.title(), "#C9B79C"))

        left, right = st.columns([1.1, 1])

        with left:
            render(
                f"""
                <div class="result" style="background:{color}55;">
                <div class="lbl">PREDICTED GENRE</div>
                <div class="name">{name}</div>
                </div>
                """
            )

        with right:
            bars = ""
            for g, score in ranked[:3]:
                g_name, g_color = GENRES.get(g, (g.title(), "#C9B79C"))
                pct = score * 100
                bars += f"""
                <div class="bar-row">
                <div class="bar-name">{g_name}</div>
                <div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%; background:{g_color};"></div></div>
                <div class="bar-val">{pct:.0f}%</div>
                </div>
                """
            render(
                f"""
                <div class="card" style="margin-top:1.2rem;">
                <h4>Top 3 model scores</h4>
                {bars}
                <div class="note">These are model scores, not guaranteed probabilities.</div>
                </div>
                """
            )

        pills = "".join(
            f'<span class="pill {"on" if key == genre else ""}" '
            f'style="{"background:" + c + "88;" if key == genre else ""}">{n}</span>'
            for key, (n, c) in GENRES.items()
        )
        render(f'<div class="pills">{pills}</div>')


# ======================================================================
# Page: Dashboard
# ======================================================================
def page_dashboard():
    render(
        """
        <div class="eyebrow">Evaluation</div>
        <h1 style="margin:0 0 1rem 0;">Model dashboard</h1>
        """
    )

    ev = get_evaluation()
    if ev is None:
        st.info("Run `python src/extract_features.py` and `python src/train_model.py` first.")
        return

    macro_f1 = ev["report"]["macro avg"]["f1-score"]
    stats = [
        (ev["model_name"], "Selected model"),
        (f"{ev['accuracy'] * 100:.1f}%", "Test accuracy"),
        (f"{macro_f1:.2f}", "Macro F1-score"),
        (f"{ev['n_test']}", "Unseen test songs"),
    ]
    cols = st.columns(4)
    for col, (num, lbl) in zip(cols, stats):
        with col:
            render(f'<div class="card"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>')

    left, right = st.columns(2)

    # --- F1 per genre ---
    with left:
        render('<div class="section-title">F1-score by genre</div>')
        f1 = {g: ev["report"][g]["f1-score"] for g in ev["classes"]}
        items = sorted(f1.items(), key=lambda x: x[1])

        fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
        ax.set_facecolor(BG)
        ax.barh(
            [GENRES[g][0] for g, _ in items],
            [v for _, v in items],
            color=[GENRES[g][1] for g, _ in items],
            height=0.65,
        )
        for i, (_, v) in enumerate(items):
            ax.text(v + 0.015, i, f"{v:.2f}", va="center", color=MUTED, fontsize=9)
        ax.set_xlim(0, 1.05)
        style_axes(ax)
        ax.set_xticks([])
        st.pyplot(fig, clear_figure=True)

    # --- Confusion matrix ---
    with right:
        render('<div class="section-title">Confusion matrix</div>')
        cmap = LinearSegmentedColormap.from_list("sage", ["#FFFDF9", "#D5DEC6", "#7C8F6B"])
        labels = [GENRES[g][0] for g in ev["classes"]]

        fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor=BG)
        sns.heatmap(
            ev["cm"], annot=True, fmt="d", cmap=cmap, cbar=False,
            xticklabels=labels, yticklabels=labels,
            linewidths=2, linecolor=BG, ax=ax,
            annot_kws={"color": INK, "fontsize": 9},
        )
        ax.set_xlabel("Predicted", color=MUTED)
        ax.set_ylabel("Actual", color=MUTED)
        ax.tick_params(colors=MUTED, length=0)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        st.pyplot(fig, clear_figure=True)

    render(
        """
        <div class="note">Each row is the true genre and each column is what the model predicted.
        Numbers off the diagonal show which genres get confused with each other.</div>
        """
    )


# ======================================================================
# Page: About
# ======================================================================
def page_about():
    render(
        """
        <div class="eyebrow">About</div>
        <h1 style="margin:0 0 1rem 0;">How this project works</h1>
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        render(
            """
            <div class="card">
            <h4>The pipeline</h4>
            <p>Audio file → Librosa loads 30 seconds → 13 MFCC, 12 chroma, spectral centroid,
            bandwidth, rolloff and zero crossing rate (29 features) → StandardScaler →
            SVM or Random Forest → predicted genre.</p>
            </div>
            """
        )
    with c2:
        render(
            """
            <div class="card">
            <h4>The dataset</h4>
            <p>GTZAN Music Genre Dataset: about 1,000 tracks across 10 genres.
            80% is used for training and 20% is kept unseen for testing,
            with the same genre proportions in both.</p>
            </div>
            """
        )

    render('<div class="section-title">Tech stack</div>')
    stack = ["Python", "Librosa", "NumPy", "Pandas", "Scikit-learn", "Matplotlib", "Seaborn", "Joblib", "Streamlit"]
    pills = "".join(f'<span class="pill">{s}</span>' for s in stack)
    render(f'<div class="pills">{pills}</div>')

    render('<div class="section-title">Limitations</div>')
    render(
        """
        <div class="card">
        <p>GTZAN contains older recordings, so modern productions can be misclassified.
        Genres such as country, disco and hip-hop overlap with others in feature space.
        The model only sees averaged audio features, not lyrics or song structure.
        Percentages shown on the Classify page are model scores, not calibrated probabilities.</p>
        </div>
        """
    )


# ======================================================================
# Router
# ======================================================================
if page == "Home":
    page_home()
elif page == "Classify":
    page_classify()
elif page == "Dashboard":
    page_dashboard()
else:
    page_about()