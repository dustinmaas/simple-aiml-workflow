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
import inspect
import tempfile
# Add the app directory to the Python path so we can import lib
sys.path.append('/app')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_SERVER_URL = os.environ.get('MODEL_SERVER_URL', 'http://model-server:5000')
DEFAULT_MODEL_ID = 'latest'

app = Flask(__name__)

# In-memory model cache
model_cache = {}

def get_model(model_id):
    try:
        loaded_model = model_cache[model_id]
    except KeyError:
        logger.info(f"Model {model_id} not found in cache, fetching from model server")
        try:
            response = requests.get(f"{MODEL_SERVER_URL}/models/{model_id}")
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching model {model_id} from model server: {e}")
            raise Exception(f"Error fetching model {model_id} from model server: {str(e)}")
        
        # Create temp file to save the downloaded model
        temp_dir = tempfile.mkdtemp()
        temp_model_path = os.path.join(temp_dir, f"{model_id}.pt")
    
        # Save the model to the temp file
        with open(temp_model_path, 'wb') as f:
            f.write(response.content)
    
        # Load the checkpoint with the model
        loaded_model = torch.jit.load(temp_model_path)
        
        # Save the model to the cache
        model_cache[model_id] = loaded_model
        
    return loaded_model

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
        # Convert features to tensor
        logger.info(f"Features: {features}")
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            logger.info(f"Model data: {model_data}")
            model, x_mean, x_std, y_mean, y_std = model_data
            
            # Normalize features if we have normalization parameters
            if x_mean is not None and x_std is not None:
                features_scaled = (features_tensor - x_mean) / x_std
            else:
                features_scaled = features_tensor
            
            # Make prediction with model
            try:
                # First try with denormalize=True
                predictions = model(features_scaled, denormalize=True)
            except Exception as e:
                # If that fails, try without denormalize parameter
                try:
                    predictions_scaled = model(features_scaled)
                    # Manually denormalize if needed
                    if y_mean is not None and y_std is not None:
                        predictions = predictions_scaled * y_std + y_mean
                    else:
                        predictions = predictions_scaled
                except Exception as inner_e:
                    raise Exception(f"Failed to make prediction: {str(e)}. Also tried without denormalize: {str(inner_e)}")
        
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