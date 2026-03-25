import pandas as pd
import numpy as np
from scipy.fft import fft

# ----------------------------
# 3. Feature Engineering
# ----------------------------


def rolling_features(X, window=5):
    n_samples, seq_len, n_sensors = X.shape
    features = []
    for i in range(n_samples):
        sample_feats = []
        for s in range(n_sensors):
            sensor_series = pd.Series(X[i,:,s])
            sample_feats.extend(sensor_series.rolling(window).mean().fillna(0).values)
            sample_feats.extend(sensor_series.rolling(window).std().fillna(0).values)
            fft_vals = np.abs(fft(sensor_series))[:seq_len]
            sample_feats.extend(fft_vals)
        features.append(sample_feats)
    return np.array(features)