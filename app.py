import streamlit as st
import joblib
import numpy as np

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="EmotionAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# LOAD MODEL
# =========================================================

MODEL_FILE = "lr_model.pkl"
VECTORIZER_FILE = "tfidf_vectorize.pkl"

try:
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
except Exception as e:
    st.error("❌ Model files load nahi ho rahe.")
    st.code(str(e))
    st.stop()


# =========================================================
# EMOTION MAPPING
# =========================================================

EMOTIONS = {
    0: ("Sadness", "😢", "You seem a little sad."),
    1: ("Joy", "😊", "You are feeling happy!"),
    2: ("Love", "❤️", "You are feeling loving."),
    3: ("Anger", "😡", "You seem angry."),
    4: ("Fear", "😨", "You seem a little scared."),
    5: ("Surprise", "😲", "You are surprised!")
}


# =========================================================
# SESSION STATE
# =========================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "text" not in st.session_state:
    st.session_state.text = ""


# =========================================================
# HEADER
# =========================================================

st.title("🧠 EmotionAI")

st.caption(
    "AI-powered emotion detection using Natural Language Processing & Machine Learning"
)

st.divider()


# =========================================================
# TOP INFO
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("🤖 Model", "Logistic Regression")

with c2:
    st.metric("🔤 NLP", "TF-IDF")

with c3:
    st.metric("🎯 Classes", "6 Emotions")

with c4:
    st.metric("⚡ Prediction", "Instant")


st.write("")


# =========================================================
# MAIN AREA
# =========================================================

left, right = st.columns([1.15, 0.85], gap="large")


# =========================================================
# LEFT SIDE - INPUT
# =========================================================

with left:

    st.subheader("💬 Tell me how you feel")

    st.caption(
        "Write anything — your model will analyze the text and detect the emotion."
    )

    user_text = st.text_area(
        "Your message",
        value=st.session_state.text,
        height=190,
        placeholder=(
            "Example:\n"
            "I am really happy today because I got my dream job!"
        ),
        label_visibility="collapsed"
    )

    st.write("")

    predict_col, clear_col = st.columns([3, 1])

    with predict_col:

        predict_button = st.button(
            "✨  Detect My Emotion",
            type="primary",
            use_container_width=True
        )

    with clear_col:

        clear_button = st.button(
            "🗑️ Clear",
            use_container_width=True
        )


# =========================================================
# CLEAR
# =========================================================

if clear_button:

    st.session_state.prediction = None
    st.session_state.confidence = None
    st.session_state.text = ""

    st.rerun()


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    if not user_text.strip():

        st.warning("⚠️ Pehle kuch text likho.")

    else:

        try:

            # TF-IDF transformation
            text_vector = vectorizer.transform([user_text])

            # Prediction
            prediction = model.predict(text_vector)[0]

            # Probability
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(text_vector)[0]
                confidence = float(np.max(probabilities)) * 100

            else:

                probabilities = None
                confidence = None

            # Save state
            st.session_state.prediction = int(prediction)
            st.session_state.confidence = confidence
            st.session_state.text = user_text

            # Celebration 🎉
            st.balloons()

            st.toast(
                "✨ Emotion successfully detected!",
                icon="🧠"
            )

        except Exception as e:

            st.error("❌ Prediction mein error aa gaya.")
            st.code(str(e))


# =========================================================
# RIGHT SIDE - RESULT
# =========================================================

with right:

    st.subheader("🎯 Emotion Result")

    st.write("")

    if st.session_state.prediction is None:

        st.info(
            "👈 Apna message likho aur **Detect My Emotion** button dabao."
        )

        st.write("")

        st.markdown("### 🌈 Available Emotions")

        emotion_cols = st.columns(3)

        emotions = [
            ("😢", "Sadness"),
            ("😊", "Joy"),
            ("❤️", "Love"),
            ("😡", "Anger"),
            ("😨", "Fear"),
            ("😲", "Surprise")
        ]

        for i, (emoji, name) in enumerate(emotions):

            with emotion_cols[i % 3]:

                st.info(f"{emoji}\n\n**{name}**")

    else:

        prediction = st.session_state.prediction

        emotion_name, emoji, message = EMOTIONS.get(
            prediction,
            (f"Class {prediction}", "🤔", "Emotion detected.")
        )

        confidence = st.session_state.confidence

        # Big result
        st.success("✨ Emotion Detected!")

        st.write("")

        st.markdown(
            f"# {emoji} {emotion_name}"
        )

        st.markdown(
            f"### {message}"
        )

        st.write("")

        # Confidence
        if confidence is not None:

            st.metric(
                "🎯 Model Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                min(confidence / 100, 1.0)
            )

        st.write("")

        st.caption(
            "Prediction generated by your trained Logistic Regression model."
        )


# =========================================================
# ANALYSIS SECTION
# =========================================================

if st.session_state.prediction is not None:

    st.divider()

    st.subheader("📊 Prediction Analysis")

    prediction = st.session_state.prediction

    emotion_name, emoji, _ = EMOTIONS.get(
        prediction,
        (f"Class {prediction}", "🤔", "")
    )

    a1, a2, a3 = st.columns(3)

    with a1:

        st.metric(
            "Detected Emotion",
            f"{emoji} {emotion_name}"
        )

    with a2:

        if st.session_state.confidence is not None:

            st.metric(
                "Confidence",
                f"{st.session_state.confidence:.1f}%"
            )

        else:

            st.metric(
                "Confidence",
                "N/A"
            )

    with a3:

        st.metric(
            "Algorithm",
            "Logistic Regression"
        )


# =========================================================
# HOW IT WORKS
# =========================================================

with st.expander("🔍 How does EmotionAI work?"):

    st.write("""
    **Step 1 — Text Input 📝**

    You enter a sentence or message.

    **Step 2 — TF-IDF 🔤**

    The text is converted into numerical features using TF-IDF.

    **Step 3 — Logistic Regression 🤖**

    The trained machine-learning model analyzes those features.

    **Step 4 — Emotion Detection 🎯**

    The model predicts one of six emotions:

    😢 Sadness  
    😊 Joy  
    ❤️ Love  
    😡 Anger  
    😨 Fear  
    😲 Surprise
    """)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🧠 EmotionAI • NLP + TF-IDF + Logistic Regression"
)