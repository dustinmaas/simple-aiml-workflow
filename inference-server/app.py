#!/usr/bin/env python3
"""
Inference Server API for PyTorch models.

This API receives RSRP and min_prb_ratio data from the experiment-runner,
and returns PyTorch model inference results.
"""

import json
import os
import logging
from flask import Flask, request, jsonify
import requests
import torch
import numpy as np

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
        
        # Load the PyTorch model
        model = torch.load(model_path)
        model.eval()  # Set to evaluation mode
        
        # Cache the model
        model_cache[model_id] = model
        return model
    
    except Exception as e:
        logger.error(f"Error fetching model: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction with a PyTorch model."""
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    model_id = data.get('model_id', DEFAULT_MODEL_ID)
    rsrp = data.get('rsrp')
    min_prb_ratio = data.get('min_prb_ratio')
    
    if rsrp is None or min_prb_ratio is None:
        return jsonify({"error": "Missing required parameters: rsrp, min_prb_ratio"}), 400
    
    # Get the model
    model = get_model(model_id)
    if model is None:
        return jsonify({"error": f"Model {model_id} not found"}), 404
    
    try:
        # Prepare input tensor
        input_tensor = torch.tensor([[float(rsrp), float(min_prb_ratio)]], dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            
        # Convert output to Python native type
        prediction = output.item() if output.numel() == 1 else output.tolist()
        
        return jsonify({
            "model_id": model_id,
            "input": {
                "rsrp": rsrp,
                "min_prb_ratio": min_prb_ratio
            },
            "prediction": prediction
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True) 