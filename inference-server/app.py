#!/usr/bin/env python3
"""
Inference Server API for PyTorch models.

This API receives feature data from clients and returns PyTorch model inference results.
"""

import json
import os
import sys
import logging
from flask import Flask, request, jsonify
import requests
import torch
import numpy as np

# Add the app directory to the Python path so we can import lib
sys.path.append('/app')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the model class
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)  # two input feature, one output feature
        self.register_buffer('x_mean', torch.zeros(2))
        self.register_buffer('x_std', torch.ones(2))
        self.register_buffer('y_mean', torch.zeros(1))
        self.register_buffer('y_std', torch.ones(1))

    def forward(self, x, denormalize=False):
        x_scaled = (x - self.x_mean) / self.x_std
        output = self.linear(x_scaled)
        if denormalize:
            output = output * self.y_std + self.y_mean
        return output

# Constants
MODEL_SERVER_URL = os.environ.get('MODEL_SERVER_URL', 'http://model-server:5000')
DEFAULT_MODEL_ID = 'latest'

app = Flask(__name__)

# In-memory model cache
model_cache = {}

def get_model(model_id):
    """Retrieve a model from the model-server or from cache."""
    if model_id in model_cache:
        logger.info(f"Using cached model: {model_id}")
        return model_cache[model_id]
    
    try:
        logger.info(f"Fetching model {model_id} from model server")
        response = requests.get(f"{MODEL_SERVER_URL}/models/{model_id}")
        response.raise_for_status()
        
        # Save the model to a temporary file
        model_path = f"/tmp/model_{model_id}.pt"
        with open(model_path, 'wb') as f:
            f.write(response.content)
        
        # Load the checkpoint
        checkpoint = torch.load(model_path)
        logger.info(f"Loaded checkpoint keys: {list(checkpoint.keys())}")
        
        # Extract model and normalization parameters
        if 'model' in checkpoint:
            # TorchScript model
            model = checkpoint['model']
            x_mean = checkpoint.get('x_mean', torch.zeros(2))
            x_std = checkpoint.get('x_std', torch.ones(2))
            y_mean = checkpoint.get('y_mean', torch.zeros(1))
            y_std = checkpoint.get('y_std', torch.ones(1))
            
            # Cache the model with normalization parameters
            model_data = (model, x_mean, x_std, y_mean, y_std)
            model_cache[model_id] = model_data
            logger.info(f"Loaded TorchScript model for {model_id}")
            return model_data
        else:
            logger.error(f"Unsupported model format: {list(checkpoint.keys())}")
            return None
        
    except Exception as e:
        logger.error(f"Error fetching model: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using locally loaded models"""
    # Parse request data
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    # Validate request data
    if 'features' not in data:
        return jsonify({"error": "Missing 'features' in request data"}), 400
    
    # Get model_id or use default
    model_id = data.get('model_id', DEFAULT_MODEL_ID)
    features = data['features']
    
    # Get or load the model
    model_data = get_model(model_id)
    if model_data is None:
        return jsonify({"error": f"Failed to load model {model_id}"}), 500
    
    try:
        model, x_mean, x_std, y_mean, y_std = model_data
        
        # Convert features to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Normalize inputs
        features_scaled = (features_tensor - x_mean) / x_std
        
        # Make prediction
        with torch.no_grad():
            predictions_scaled = model(features_scaled)
            # Denormalize
            predictions = predictions_scaled * y_std + y_mean
        
        # Convert predictions to list
        predictions = predictions.squeeze().tolist()
        
        # Handle single prediction vs multiple predictions
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        return jsonify({
            "predictions": predictions,
            "model_id": model_id
        }), 200
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({"error": f"Error making prediction: {str(e)}"}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models from the model server"""
    try:
        response = requests.get(f"{MODEL_SERVER_URL}/models", timeout=10)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting model list from model server: {e}")
        return jsonify({"error": f"Error communicating with model server: {str(e)}"}), 500

@app.route('/predict/prb', methods=['POST'])
def predict_prb():
    """Specialized endpoint for PRB prediction based on CQI and throughput"""
    # Parse request data
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    # Validate request data
    if 'cqi' not in data:
        return jsonify({"error": "Missing 'cqi' in request data"}), 400
    
    if 'throughput' not in data:
        return jsonify({"error": "Missing 'throughput' in request data"}), 400
    
    # Get model_id or use default
    model_id = data.get('model_id', 'torch_linear_regression_v1')
    
    # Prepare features
    cqi = data['cqi']
    throughput = data['throughput']
    
    # Handle both single value and list inputs
    if isinstance(cqi, (int, float)) and isinstance(throughput, (int, float)):
        features = [[float(cqi), float(throughput)]]
    elif isinstance(cqi, list) and isinstance(throughput, list) and len(cqi) == len(throughput):
        features = [[float(c), float(t)] for c, t in zip(cqi, throughput)]
    else:
        return jsonify({"error": "CQI and throughput must be either single values or lists of the same length"}), 400
    
    # Forward the prediction request to the model server
    try:
        response = requests.post(
            f"{MODEL_SERVER_URL}/models/{model_id}/predict",
            json={"features": features},
            timeout=30
        )
        
        if response.status_code != 200:
            return jsonify(response.json()), response.status_code
        
        # Extract predictions and format response
        result = response.json()
        predictions = result.get('predictions', [])
        
        # Format response based on input type
        if len(features) == 1:
            return jsonify({
                "min_prb_ratio": predictions[0],
                "model_id": model_id
            }), 200
        else:
            return jsonify({
                "min_prb_ratio": predictions,
                "model_id": model_id
            }), 200
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error forwarding request to model server: {e}")
        return jsonify({"error": f"Error communicating with model server: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 