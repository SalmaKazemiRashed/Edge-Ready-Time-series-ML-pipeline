import utils
import data_ingestion
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.fft import fft
from feauture_engineering import rolling_features

# ----------------------------
# 4. Classical ML Training
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))