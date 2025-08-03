# train.py

from data_preprocessing import load_and_preprocess
from model import build_autoencoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model():
    # Load and preprocess data
    X, y = load_and_preprocess()

    # Use only non-fraud cases for training
    X_train = X[y == 0]

    # Build the autoencoder
    autoencoder = build_autoencoder(input_dim=X.shape[1])

    # Add callbacks for saving and early stopping
    checkpoint = ModelCheckpoint(
        filepath='autoencoder_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train the model
    history = autoencoder.fit(
        X_train, X_train,
        epochs=20,
        batch_size=512,
        shuffle=True,
        validation_split=0.1,
        verbose=1,
        callbacks=[checkpoint, early_stop]
    )

    print("✅ Model training completed and saved as autoencoder_model.h5")

if __name__ == "__main__":
    train_model()
