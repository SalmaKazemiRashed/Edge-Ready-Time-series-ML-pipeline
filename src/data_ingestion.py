import numpy as np

# ----------------------------
# 1. Data Ingestion & Simulation
# ---------------------------

def generate_sensor_data(n_samples=1000, seq_len=50, n_sensors=3):
    data = np.random.normal(0, 1, (n_samples, seq_len, n_sensors))
    # Inject anomalies
    for _ in range(int(0.05 * n_samples)):
        idx = np.random.randint(0, n_samples)
        sensor_idx = np.random.randint(0, n_sensors)
        data[idx, :, sensor_idx] += np.random.normal(5, 1, seq_len)
    return data

def create_labels(data, threshold=3):
    labels = np.zeros(len(data))
    for i, sample in enumerate(data):
        if np.any(np.abs(sample) > threshold):
            labels[i] = 1
    return labels