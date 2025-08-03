import numpy as np

def detect_anomalies(model, X, y):
    reconstructions = model.predict(X)
    mse = np.mean(np.power(X - reconstructions, 2), axis=1)
    
    threshold = np.percentile(mse[y == 0], 95)
    y_pred = mse > threshold
    
    return y_pred, mse, threshold
