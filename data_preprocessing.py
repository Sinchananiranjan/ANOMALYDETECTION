import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    df = pd.read_csv('data/creditcard.csv')

    # Separate features and labels
    X = df.drop(['Class'], axis=1)
    y = df['Class']

    # Normalize 'Time' and 'Amount'
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
    
    return X, y
