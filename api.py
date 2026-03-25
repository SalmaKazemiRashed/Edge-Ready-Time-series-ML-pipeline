# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI()

# -----------------------------
# 1. Request model
# -----------------------------
class SensorDataRequest(BaseModel):
    sensor_data: list  # List of [timesteps, sensors]

# -----------------------------
# 2. Load TFLite model
# -----------------------------
interpreter = tf.lite.Interpreter(model_path="edge_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# 3. Prediction function
# -----------------------------
def predict(sensor_data, seq_len=50):
    # Convert to numpy
    sensor_data = np.array(sensor_data, dtype=np.float32)

    # Pad sequence if too short
    if sensor_data.shape[0] < seq_len:
        pad_len = seq_len - sensor_data.shape[0]
        padding = np.repeat(sensor_data[-1][np.newaxis, :], pad_len, axis=0)
        sensor_data = np.vstack([sensor_data, padding])

    # Add batch dimension: (1, seq_len, n_sensors)
    sensor_data = sensor_data[np.newaxis, :, :]

    # Set tensor and invoke
    interpreter.set_tensor(input_details[0]['index'], sensor_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction.tolist()  # convert to list for JSON

# -----------------------------
# 4. FastAPI endpoint
# -----------------------------
@app.post("/predict")
def predict_endpoint(request: SensorDataRequest):
    try:
        prediction = predict(request.sensor_data)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}