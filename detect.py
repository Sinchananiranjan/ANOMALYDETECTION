from tensorflow.keras.models import load_model
from data_preprocessing import load_and_preprocess
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 👇 Fix: load model without compiling
model = load_model('autoencoder_model.h5', compile=False)

# Load and preprocess data
X, y = load_and_preprocess()

# Get reconstruction
reconstructions = model.predict(X)
mse = np.mean(np.square(X - reconstructions), axis=1)

# Choose a threshold
threshold = np.percentile(mse[y == 0], 95)

# Classify
y_pred = (mse > threshold).astype(int)

# Show metrics
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred))

# Optional: Show error plot
plt.hist(mse[y == 0], bins=50, alpha=0.6, label='Normal')
plt.hist(mse[y == 1], bins=50, alpha=0.6, label='Fraud')
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.title("Reconstruction Error")
plt.xlabel("MSE")
plt.ylabel("Count")
plt.legend()
plt.show()
