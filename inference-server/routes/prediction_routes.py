#!/usr/bin/env python3
"""
Prediction routes for the inference server.

This module defines routes for making predictions using ONNX models.
"""

import os
import logging
import json
import requests
import numpy as np
from typing import Dict, Any, List, Optional
from flask import Blueprint, request, jsonify, current_app

from utils.constants import DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION

logger = logging.getLogger(__name__)

# Create Blueprint for prediction routes
prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions using versioned ONNX models.
    
    Request body should contain:
    - features: List of input features
    - model_name: (optional) Name of the model to use
    - model_version: (optional) Version of the model to use
    
    Returns:
        JSON response with prediction results and model information
    """
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
        
        # Get model and metadata from cache
        session, metadata = current_app.model_cache.get_model(model_name, model_version)
        
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

@prediction_bp.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Make predictions for multiple batches using versioned ONNX models.
    
    Request body should contain:
    - batches: List of feature batches, each a list of feature vectors
    - model_name: (optional) Name of the model to use
    - model_version: (optional) Version of the model to use
    
    Returns:
        JSON response with prediction results for each batch and model information
    """
    # Parse request data
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    # Validate request data
    if 'batches' not in data:
        return jsonify({"error": "Missing 'batches' in request data"}), 400
    
    # Get model information or use defaults
    model_name = data.get('model_name', DEFAULT_MODEL_NAME)
    model_version = data.get('model_version', DEFAULT_MODEL_VERSION)
    batches = data['batches']
    
    # Validate batches
    if not isinstance(batches, list):
        return jsonify({"error": "Batches must be a list of lists"}), 400
    
    if len(batches) == 0:
        return jsonify({"error": "Batches list cannot be empty"}), 400
        
    try:
        # Get model and metadata from cache
        session, metadata = current_app.model_cache.get_model(model_name, model_version)
        input_name = session.get_inputs()[0].name
        
        # Process each batch
        results = []
        for i, batch in enumerate(batches):
            try:
                # Convert to NumPy array
                features_array = np.array(batch, dtype=np.float32)
                
                # Make prediction
                outputs = session.run(None, {input_name: features_array})
                
                # Convert to list and add to results
                predictions = outputs[0].tolist()
                results.append({
                    "batch_index": i,
                    "predictions": predictions,
                    "success": True
                })
            except Exception as batch_error:
                # Record error for this batch but continue processing
                results.append({
                    "batch_index": i,
                    "error": str(batch_error),
                    "success": False
                })
        
        # Construct response with model information
        response = {
            "batch_results": results,
            "success_count": sum(1 for r in results if r["success"]),
            "error_count": sum(1 for r in results if not r["success"]),
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
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({"error": f"Error in batch prediction: {str(e)}"}), 500

@prediction_bp.route('/models', methods=['GET'])
def list_models():
    """
    List available models from the model server.
    
    Returns:
        JSON response with available models
    """
    model_server_url = current_app.config['MODEL_SERVER_URL']
    
    try:
        response = requests.get(f"{model_server_url}/models", timeout=10)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting model list from model server: {e}")
        return jsonify({"error": f"Error communicating with model server: {str(e)}"}), 500
        
@prediction_bp.route('/models/<model_name>/versions', methods=['GET'])
def list_model_versions(model_name):
    """
    List available versions of a specific model.
    
    Args:
        model_name: Base name of the model
        
    Returns:
        JSON response with available versions of the model
    """
    model_server_url = current_app.config['MODEL_SERVER_URL']
    
    try:
        response = requests.get(f"{model_server_url}/models/{model_name}/versions", timeout=10)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting model versions from model server: {e}")
        return jsonify({"error": f"Error communicating with model server: {str(e)}"}), 500
