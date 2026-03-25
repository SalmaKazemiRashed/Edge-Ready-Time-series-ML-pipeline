import tensorflow as tf
import time
import numpy as np

# ----------------------------
# 6. Edge Deployment (TFLite)
# ----------------------------

def convert_to_tflite(model, path="edge_model.tflite"):
    tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    with open(path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved: {path}")
    return path

def benchmark_tflite(tflite_path, sample_input):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    sample_input = np.array(sample_input, dtype=np.float32)
    sample_input = np.expand_dims(sample_input, axis=0)
    interpreter.set_tensor(input_details[0]['index'], sample_input)
    start = time.time()
    interpreter.invoke()
    end = time.time()
    pred = interpreter.get_tensor(output_details[0]['index'])
    print(f"TFLite Inference Result: {pred}, Latency: {(end-start)*1000:.2f} ms")
    return pred