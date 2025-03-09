#!/usr/bin/env python3
"""
Inference Server API for ONNX models.

This API receives feature data from clients and returns inference results using versioned ONNX models.
"""

import json
import os
import sys
import logging
import re
from flask import Flask, request, jsonify
import requests
import numpy as np
import tempfile
import onnxruntime as ort

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_SERVER_URL = os.environ.get('MODEL_SERVER_URL', 'http://model-server:5000')
DEFAULT_MODEL_NAME = 'linear_regression_model'  # Default model base name
DEFAULT_MODEL_VERSION = 'latest'  # Default to latest version

app = Flask(__name__)

# In-memory model cache - keyed by "model_name:version"
model_cache = {}

def get_model_metadata(model_name, version=None):
    """
    Get metadata for a model from the model server.
    
    Args:
        model_name: Base name of the model
        version: Specific version to retrieve, or None for latest
        
    Returns:
        Dictionary containing model metadata
    """
    try:
        version_str = version if version and version != 'latest' else 'latest'
        url = f"{MODEL_SERVER_URL}/models/{model_name}/versions/{version_str}/metadata"
        
        logger.info(f"Fetching metadata from {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching metadata for {model_name} version {version}: {e}")
        return {}  # Return empty dict if metadata can't be fetched
    except Exception as e:
        logger.warning(f"Error processing metadata: {e}")
        return {}

def get_model(model_name, version=None):
    """
    Get a model from cache or from the model server.
    
    Args:
        model_name: Base name of the model
        version: Specific version to retrieve, or None for latest
        
    Returns:
        Tuple of (ONNX Runtime InferenceSession, metadata dict)
    """
    # Create a cache key for this model and version
    cache_key = f"{model_name}:{version or 'latest'}"
    
    # Check if model is in cache
    if cache_key in model_cache:
        logger.info(f"Using cached model {cache_key}")
        return model_cache[cache_key]
        
    # Model not in cache, fetch from server
    try:
        if version == 'latest' or version is None:
            # Get the latest version
            url = f"{MODEL_SERVER_URL}/models/{model_name}/versions/latest"
        else:
            # Get specific version
            url = f"{MODEL_SERVER_URL}/models/{model_name}/versions/{version}"
            
        logger.info(f"Fetching model from {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        # Create a temporary file for the model
        temp_dir = tempfile.mkdtemp()
        extension = '.onnx'
        file_version = version or 'latest'
        temp_model_path = os.path.join(temp_dir, f"{model_name}_v{file_version}{extension}")
        
        # Save the model to a temporary file
        with open(temp_model_path, 'wb') as f:
            f.write(response.content)
            
        # Load the ONNX model
        session = ort.InferenceSession(temp_model_path)
        
        # Try to get metadata from separate endpoint first, and fall back to embedded metadata
        metadata = get_model_metadata(model_name, version)
        if not metadata:
            # If no metadata from separate endpoint, try to extract from model
            try:
                embedded_metadata = session.get_modelmeta().custom_metadata_map
                if embedded_metadata:
                    metadata = {k: v for k, v in embedded_metadata.items()}
                    logger.info(f"Using embedded metadata: {metadata}")
            except Exception as e:
                logger.warning(f"Failed to extract embedded metadata: {e}")
            
        # Clean up temporary file
        os.remove(temp_model_path)
        os.rmdir(temp_dir)
        
        # Cache the model session
        model_cache[cache_key] = (session, metadata)
        return model_cache[cache_key]
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching model {model_name} version {version}: {e}")
        raise Exception(f"Error fetching model: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise Exception(f"Error loading model: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using versioned ONNX models"""
    # Parse request data
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    # Validate request data
    if 'features' not in data:
        return jsonify({"error": "Missing 'features' in request data"}), 400
    
    # Get model information or use defaults
    model_name = data.get('model_name', DEFAULT_MODEL_NAME)
    model_version = data.get('model_version', DEFAULT_MODEL_VERSION)
    features = data['features']
    
    # Validate features
    if not isinstance(features, list):
        return jsonify({"error": "Features must be a list"}), 400
    
    if len(features) == 0:
        return jsonify({"error": "Features list cannot be empty"}), 400
        
    try:
        # Convert features to NumPy array and ensure correct dimensionality
        features_array = np.array(features, dtype=np.float32)
        
        # Check if we have a batch of inputs or single input
        if len(features_array.shape) == 1:
            # Single input, add batch dimension
            features_array = np.expand_dims(features_array, axis=0)
        
        # Get model and metadata
        session, metadata = get_model(model_name, model_version)
        
        # Make prediction with ONNX Runtime
        input_name = session.get_inputs()[0].name  # Usually "input"
        outputs = session.run(None, {input_name: features_array})
        
        # Extract predictions (outputs is a list of arrays, usually just one)
        predictions = outputs[0]
        
        # Convert to Python types for JSON serialization
        if len(predictions.shape) > 1 and predictions.shape[0] == 1:
            # If it's a single batch prediction, flatten it
            predictions_list = predictions[0].tolist()
        else:
            predictions_list = predictions.tolist()
        
        # Construct response with model information
        response = {
            "predictions": predictions_list,
            "model": {
                "name": model_name,
                "version": model_version if model_version != 'latest' else metadata.get('version', 'unknown'),
            }
        }
        
        # Add metadata to response if available
        if metadata:
            response["model"]["metadata"] = {
                "training_date": metadata.get("training_date", "unknown"),
                "description": metadata.get("description", "unknown"),
                "metrics": metadata.get("metrics", "unknown")
            }
        
        return jsonify(response), 200
        
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
        
@app.route('/models/<model_name>/versions', methods=['GET'])
def list_model_versions(model_name):
    """List available versions of a specific model"""
    try:
        response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions", timeout=10)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting model versions from model server: {e}")
        return jsonify({"error": f"Error communicating with model server: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
