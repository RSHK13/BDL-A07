import sys
import time
from typing import List
from fastapi import FastAPI, UploadFile, File, Request
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
# from keras.models import load_model
import cv2
import uvicorn
import psutil

# Create FastAPI instance
app = FastAPI()

# Setup Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Counter for tracking API usage from different client IP addresses
api_usage_counter = Counter("api_usage", "API Usage", ["client_ip"])

# Gauges for monitoring API run time, T/L time, etc.
api_run_time_gauge = Gauge("api_run_time", "API Run Time")
api_tl_time_gauge = Gauge("api_tl_time", "API T/L Time")
memory_usage = Gauge("api_memory_usage", "Memory usage of API process")
cpu_usage = Gauge("cpu_usage", "CPU usage of API")

network_bytes_sent = Gauge("network_sent", "Network bytes sent by API")
network_bytes_revc = Gauge("network_recieved", "Network bytes recieved by API")

# Function to resize and format the uploaded image
def format_image(contents: bytes) -> np.ndarray:
    # Read the image as a NumPy array using OpenCV
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    image = cv2.bitwise_not(image)  # TO reverse pixel colors for white background

    # Resize the image to 28x28
    image = cv2.resize(image, (28, 28))

    # Flatten the image into a 1D array
    formatted_image = np.array(image)
    formatted_image = formatted_image / 255.0
    formatted_image = formatted_image.reshape((1, 784))  # Reshape to (1, 784)

    return formatted_image

# Function to predict digit from the image data
def predict_digit(data_point: np.ndarray) -> str:
    # Predict the digit
    prediction = 1
    return str(prediction)

# API endpoint for digit prediction
@app.post('/predict')
async def predict(request: Request, file: UploadFile = File(...)):
    start_time = time.time()

    # Increment API usage counter for this client IP address
    client_ip = request.client.host
    api_usage_counter.labels(client_ip).inc()
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_usage.set(cpu_percent)

    memory_usage.set(psutil.virtual_memory().used/(1024**3)) 

    network_io_counters = psutil.net_io_counters()
    network_bytes_sent.set(network_io_counters.bytes_sent)
    network_bytes_revc.set(network_io_counters.bytes_recv)
    # Load the model
    # model_path = sys.argv[1]
    # model = load_model(model_path)

    # Read the uploaded image file as bytes
    contents = await file.read()
    # Preprocess the uploaded image
    formatted_image = format_image(contents)

    # Predict the digit
    digit = predict_digit(formatted_image)

    end_time = time.time()
    total_time = (end_time - start_time) * 1000  # in milliseconds

    # Update gauges
    api_run_time_gauge.set(total_time)
    api_tl_time_gauge.set(total_time / len(contents))

    return {"digit": digit}

if __name__ == "__main__":
    # Host the FastAPI module
    # Visit http://0.0.0.0:8000/docs to use the Swagger UI
    uvicorn.run(app, host="127.0.0.1", port=5000)
