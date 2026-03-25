from src.data_ingestion import generate_sensor_data, create_labels
from src.preprocessing import flatten_data, normalize_data
from src.feature_engineering import rolling_features
from src.train_classical import train_random_forest
from src.train_nn import train_cnn
from src.deploy import convert_to_tflite, benchmark_tflite
from src.visualization import plot_sample_series

# 1. Data
X = generate_sensor_data()
y = create_labels(X)

# 2. Preprocessing
X_flat = flatten_data(X)
X_flat = normalize_data(X_flat)

# 3. Feature Engineering
X_features = rolling_features(X)

# 4. Classical ML
clf = train_random_forest(X_features, y)

# 5. Neural Network
model, X_test_nn, y_test_nn = train_cnn(X, y, epochs=10)

# 6. Edge Deployment
tflite_path = convert_to_tflite(model)
benchmark_tflite(tflite_path, X_test_nn[0])

# 7. Visualization
plot_sample_series(X_test_nn[0])