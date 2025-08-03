import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("autoencoder_model.h5")

# Preprocessing function
def preprocess(df):
    df = df.copy()
    df[['Time', 'Amount']] = StandardScaler().fit_transform(df[['Time', 'Amount']])
    return df

# Predict fraud using reconstruction error
def get_predictions(model, data, threshold):
    reconstructions = model.predict(data)
    mse = np.mean(np.power(data - reconstructions, 2), axis=1)
    predictions = (mse > threshold).astype(int)
    return mse, predictions

# App UI
st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("Upload a CSV file to detect fraudulent transactions using an autoencoder model.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Validate input
    if 'Class' in df.columns:
        true_labels = df['Class']
        X = df.drop(['Class'], axis=1)
    else:
        st.warning("`Class` column not found — prediction only, no evaluation.")
        X = df

    model = load_model()
    X_processed = preprocess(X)

    # Threshold slider
    st.sidebar.subheader("Threshold Adjustment")
    threshold = st.sidebar.slider("Set MSE Threshold", min_value=0.001, max_value=1.0, value=0.3, step=0.01)

    # Prediction
    mse, preds = get_predictions(model, X_processed, threshold)
    df['Reconstruction Error'] = mse
    df['Prediction'] = preds

    st.subheader("📊 Prediction Results")
    st.write(df[['Reconstruction Error', 'Prediction']].head(10))

    # Plot error distribution
    st.subheader("🧠 Reconstruction Error Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(mse, bins=100, kde=True, ax=ax, color="blue")
    ax.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold}")
    ax.legend()
    st.pyplot(fig)

    # Evaluation (if true labels available)
    if 'Class' in df.columns:
        from sklearn.metrics import classification_report, confusion_matrix
        st.subheader("📋 Evaluation Metrics")
        st.text(classification_report(true_labels, preds))
        st.text("Confusion Matrix:")
        st.write(confusion_matrix(true_labels, preds))
