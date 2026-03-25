import numpy as np
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 2. Data Preprocessing
# ----------------------------

def flatten_data(X):
    return X.reshape(X.shape[0], -1)

def normalize_data(X_flat):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    return X_scaled