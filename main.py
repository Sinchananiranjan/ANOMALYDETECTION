from data_preprocessing import load_and_preprocess
from model import build_autoencoder
from anomaly_detection import detect_anomalies
from evaluate import evaluate
from train import autoencoder  # Assuming you return the model from train.py

X, y = load_and_preprocess()
y_pred, mse, threshold = detect_anomalies(autoencoder, X, y)
evaluate(y, y_pred)
