import utils
import data_ingestion
import numpy as np
from sklearn.preprocessing import StandardScaler

X = data_ingestion.generate_sensor_data()
y = data_ingestion.create_labels(X)
# ----------------------------
# 2. Data Preprocessing
# ----------------------------
# Flatten for classical ML
X_flat = X.reshape(X.shape[0], -1)
# Handle missing values (none in synthetic, but example)
X_flat = np.nan_to_num(X_flat)
# Normalize
scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat)

