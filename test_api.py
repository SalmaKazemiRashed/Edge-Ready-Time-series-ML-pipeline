# test_api.py
import requests
import numpy as np

# ---------------------------
# 1. Prepare sensor data
# ---------------------------
raw_sensor_data = [
    [0.1, 0.2, 0.3],
    [0.2, 0.1, 0.4],
    [0.3, 0.2, 0.5]
]

sensor_data = np.array(raw_sensor_data, dtype=np.float32)

# ---------------------------
# 2. Send POST request
# ---------------------------
url = "http://127.0.0.1:8000/predict"
payload = {"sensor_data": sensor_data.tolist()}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # raise error if status != 200
    print("Response JSON:", response.json())
except requests.exceptions.HTTPError as e:
    print("HTTP error:", e)
    print("Status code:", response.status_code)
    print("Response text:", response.text)
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
except ValueError:
    print("Server did not return valid JSON!")
    print("Status code:", response.status_code)
    print("Response text:", response.text)