# 💳 Credit Card Fraud Detection using Tabular Autoencoder

An unsupervised deep learning model for detecting fraudulent credit card transactions by identifying deviations from normal patterns.

---

## 📌 Project Overview

This project applies a **tabular autoencoder** to credit card fraud detection. Since fraudulent transactions are rare and often unpredictable, a supervised model may not generalize well. Instead, we use **unsupervised anomaly detection**—training only on legitimate transactions, and identifying fraud based on reconstruction error.

* **Dataset**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Goal**: Detect outliers (fraudulent transactions) without using fraud labels during training.
* **Approach**: Train an autoencoder to learn the structure of normal transactions. Fraud is detected when reconstruction error exceeds a certain threshold.

---

## ⚙️ How It Works

1. ### 🔄 Data Preprocessing

   * Load and clean the dataset.
   * Normalize features (`V1–V28`, `Amount`) using `StandardScaler`.
   * Split data into:

     * **Training set**: Only non-fraudulent samples (`Class = 0`)
     * **Test/validation sets**: Include both normal and fraudulent samples

2. ### 🧠 Autoencoder Model

   * A neural network consisting of:

     * **Encoder**: Compresses the input into a low-dimensional latent space.
     * **Decoder**: Attempts to reconstruct the original input from the latent space.
   * The network is trained to minimize **reconstruction error** on normal transactions.

3. ### 🚨 Anomaly Detection Logic

   * At inference, all transactions are passed through the autoencoder.
   * For each transaction, calculate the **reconstruction error** (e.g., MSE).
   * Define a **threshold**: transactions with error above this value are flagged as fraudulent.
   * Threshold can be:

     * Fixed statistically (e.g. 95th percentile of training error)
     * Tuned based on validation set performance

4. ### 📈 Model Evaluation

   * Evaluate the predictions using true fraud labels (for validation only):

     * **Precision**, **Recall**, **F1-Score**
     * **ROC-AUC**
     * **Confusion Matrix**
   * Visualization of reconstruction error distributions helps understand model behavior.

---

## 💡 Why This Works

* **Autoencoders learn the structure of normal data**. Anything that deviates from this (like fraud) will reconstruct poorly.
* No need to rely on fraud labels during training—making this approach practical for real-world systems where fraud evolves and labels are delayed or missing.
* This method reinforces the idea that **anomalies are relative to the normal patterns the model learns**.

---
