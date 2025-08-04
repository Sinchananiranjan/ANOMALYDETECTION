Certainly! Here's the updated and polished version of your project README with a **direct Kaggle link** and a clear explanation of how the fraud detection works using an autoencoder:

---

# 💳 Credit Card Fraud Detection using Tabular Autoencoder

An unsupervised deep learning model to detect fraudulent credit card transactions by identifying anomalies in transaction behavior.

---

## 📌 Project Overview

This project aims to detect credit card fraud using a **tabular autoencoder**—a neural network that learns normal transaction patterns and flags suspicious activity based on reconstruction error.

* **Dataset**: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Goal**: Identify anomalies (fraudulent transactions) without needing fraud labels during training.
* **Method**: Train an autoencoder only on legitimate transactions. Use reconstruction error to identify fraud.

---

## ⚙️ How It Works

1. **🔍 Data Preprocessing**

   * Load the dataset from Kaggle (CSV format).
   * Normalize features using `StandardScaler` for stability.
   * Use only **non-fraudulent transactions** (`Class == 0`) for training.
   * Fraudulent transactions (`Class == 1`) are reserved for validation and testing.

2. **🧠 Autoencoder Architecture**

   * A neural network with:

     * **Encoder**: Compresses the input into a latent representation.
     * **Decoder**: Reconstructs the input from the latent space.
   * Trained to minimize reconstruction error (Mean Squared Error).

3. **🚨 Fraud Detection Logic**

   * After training, use the model to reconstruct both normal and suspicious transactions.
   * Calculate reconstruction error for each transaction.
   * Set a **threshold** based on the error distribution or using validation performance.
   * Flag any transaction with error above the threshold as **potential fraud**.

4. **📈 Evaluation Metrics**

   * Compare predictions to true labels using:

     * **Precision**, **Recall**, **F1-Score**
     * **ROC-AUC Score**
     * **Confusion Matrix**
   * Visualize reconstruction error distributions for normal vs. fraud cases.

---

## 📊 Why Use Autoencoders?

* Handles imbalanced data naturally (no need for oversampling).
* Detects unseen or evolving fraud patterns.
* Works in unsupervised or semi-supervised settings where labels are limited or delayed.

---

