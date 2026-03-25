# Edge-Ready-Time-series-ML-pipeline
A demo of an end-to-end pipeline for time-series sensor data analysis:

## Project OverView
´
* Synthetic data generation, cleaning, normalization, outlier removal

* Feature engineering: rolling statistics + FFT

* Classical ML + Neural Networks (1D-CNN & LSTM)

* Edge deployment via TFLite & ONNX with quantization (light models)

* Visualization of data and model predictions 

* Optional FastAPI for hybrid edge-cloud simulation (Future steps)

The File structure would be 

```plaintext
edge_sensor_pipeline/
├── data/                        
├── notebooks/                   # Jupyter notebooks for EDA
├── src/
│   ├── data_ingestion.py        # Generate/load sensor data
│   ├── preprocessing.py         # Cleaning, normalization, outlier removal
│   ├── feature_engineering.py   # Rolling stats, FFT features
│   ├── train_classical.py       # Classical ML models 
│   ├── train_nn.py              # 1D-CNN & LSTM training
│   ├── deploy.py                # TFLite/ONNX conversion, quantization
│   ├── visualize.py             # Matplotlib dashboards
│   └── utils.py                 # Helper functions
├── docker/                      # Dockerfile 
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview, instructions
└── main.py                      # Run full pipeline end-to-end
```


For running locally:
```python
python main.py
```

The [tflite model](edge_model.tflite) is saved and the results are visualized ![](static/Figure_1.png)


Now, one step further we took and want to make this pipeline production-ready. 
We make a clean FastAPI + Docker setup that turns the ML pipeline into a hybrid edge-cloud service.

We have defined a FASTAPI Inference Service where we :

* Load trained model (TFLite model)
* Accept sensor data
* Return anomaly predictions
* Simulate edge-cloud interaction

We have started API with 
```bash
uvicorn api:app --reload
```


and test the API with the [python code](test_api.py) in a separate terminal of :
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"sensor_data": [[0.1,0.2,0.3],[0.2,0.1,0.4],[0.3,0.2,0.5]]}'
```


Also, for buiding the Docker image of this app, we run Docker Desktop in windows (install first)
and then
```bash
 docker build -t edge-ml-api .
```

The optimized dockerfile:
```dockerfile
# Use a small Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency list first (cached layer)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

after building the image (edge-ml-api) we can run it through 
```bash
docker run -p 8000:8000 edge-ml-api
```