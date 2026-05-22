import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #f5f7fa, #e4ecf7);
}

/* Card-like containers */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Title styling */
h1 {
    color: #2c3e50;
    text-align: center;
}

/* Button styling */
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 200px;
    font-size: 16px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #45a049;
}

/* Text area styling */
textarea {
    border-radius: 10px !important;
    border: 1px solid #ccc !important;
}
</style>
""", unsafe_allow_html=True)


# Load Models
# -------------------------------
lgbm = joblib.load("lgbm_full.joblib")
meta = joblib.load("meta_clf.joblib")

cnn = tf.keras.models.load_model("cnn_full.keras", compile=False)
lstm = tf.keras.models.load_model("lstm_full.keras", compile=False)
bilstm = tf.keras.models.load_model("bilstm_full.keras", compile=False)

# -------------------------------
# UI CONFIG
# -------------------------------
st.set_page_config(page_title="Stress Detection", layout="centered")

st.title("🧠 Hybrid Stress Detection System")
st.write("Enter **48 physiological feature values** (comma-separated):")

input_text = st.text_area("Input Features")

show_details = st.checkbox("📊 Show Model Probabilities")

# -------------------------------
# MAIN LOGIC
# -------------------------------
if st.button("Detect Stress"):
    try:
        #  CLEAN INPUT
        raw = input_text.replace("\n", "").replace(" ", "")
        values = raw.split(",")
        clean_values = [v for v in values if v != ""]

        #  VALIDATION
        if len(clean_values) != 48:
            st.error(f"❌ Please enter exactly 48 values (You entered {len(clean_values)})")
            st.stop()

        #  CONVERT
        data = np.array([float(v) for v in clean_values]).reshape(1, -1)

        st.write(f"📥 Input shape: {data.shape}")

        # -------------------------------
        # MODEL DETECTION
        # -------------------------------
        with st.spinner("🧠 Detecting stress... please wait ⏳"):

            # LightGBM
            lgb_p = lgbm.predict_proba(data)

            # IMPORTANT reshape (fix for CNN/LSTM)
            seq_data = data.reshape(1, 12, 4)

            # Deep models
            cnn_p = cnn.predict(seq_data, verbose=0)
            lstm_p = lstm.predict(seq_data, verbose=0)
            bilstm_p = bilstm.predict(seq_data, verbose=0)

        st.success("🧠 Detection complete ✅")

        # -------------------------------
        # ENSEMBLE
        # -------------------------------
        final_proba = (lgb_p + cnn_p + lstm_p + bilstm_p) / 4
        final_pred = np.argmax(final_proba)
        confidence = np.max(final_proba)
        confidence_percent = round(confidence * 100, 2)

        # -------------------------------
        # OUTPUT UI
        # -------------------------------
        progress = st.progress(0)

        if final_pred == 0:
           st.success(f"🟢 Low Stress ({confidence_percent}% confidence)")
           progress.progress(30)

        elif final_pred == 1:
           st.warning(f"🟡 Medium Stress ({confidence_percent}% confidence)")
           progress.progress(60)

        else:
           st.error(f"🔴 High Stress ({confidence_percent}% confidence)")
           progress.progress(90)

        st.subheader("Confidence Level")
        st.progress(int(confidence_percent))

        # -------------------------------
        # DETAILS
        # -------------------------------
        if show_details:
            st.subheader("📊 Model Probabilities")
            st.write(f"LightGBM: {np.round(lgb_p, 3)}")
            st.write(f"CNN: {np.round(cnn_p, 3)}")
            st.write(f"LSTM: {np.round(lstm_p, 3)}")
            st.write(f"BiLSTM: {np.round(bilstm_p, 3)}")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Developed by Swaraj | Hybrid AI Stress Detection System")