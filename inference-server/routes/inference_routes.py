#!/usr/bin/env python3
"""
Inference routes for the inference server.

This module defines routes for model inference operations including:
- Getting predictions from models by UUID, name/version, or latest version
- Model metadata retrieval
- Model cache management
"""

import os
import logging
import json
import time
from typing import Dict, Any, Optional
import onnxruntime as ort
from flask import Blueprint, request, jsonify, current_app, Response
import numpy as np

from utils.constants import REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

# Create Blueprint for inference routes
inference_bp = Blueprint('inference', __name__, url_prefix='/inference')

@inference_bp.route('/models/<model_name>/versions/<version>/predict', methods=['POST'])
def predict_by_name_version(model_name: str, version: str):
    """
    Make a prediction using a specific model version.
    
    Args:
        model_name: Name of the model
        version: Specific version to use
        
    Returns:
        JSON response with prediction results
    """
    try:
        # Get the model file and metadata
        start_time = time.time()
        model_path, metadata = current_app.model_service.get_model_by_name_version(model_name, version)
        
        if not model_path:
            return jsonify({"error": f"Model {model_name} version {version} not found or could not be loaded"}), 404
        
        # Get input data from request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        input_data = request.get_json()
        
        # Run prediction
        result = run_prediction(model_path, input_data)
        
        # Format response
        processing_time = time.time() - start_time
        response = {
            "model_name": model_name,
            "model_version": version,
            "prediction": result,
            "processing_time_seconds": processing_time
        }
        
        # Include metadata if available
        if metadata:
            response["metadata"] = metadata
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error making prediction with {model_name} version {version}: {e}")
        return jsonify({"error": str(e)}), 500

@inference_bp.route('/models/<model_name>/latest/predict', methods=['POST'])
def predict_latest_version(model_name: str):
    """
    Make a prediction using the latest version of a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        JSON response with prediction results
    """
    try:
        # Get the latest model version
        start_time = time.time()
        model_path, metadata = current_app.model_service.get_latest_model_version(model_name)
        
        if not model_path:
            return jsonify({"error": f"No versions found for model {model_name} or could not be loaded"}), 404
        
        # Get input data from request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        input_data = request.get_json()
        
        # Run prediction
        result = run_prediction(model_path, input_data)
        
        # Get version from metadata if available
        version = "unknown"
        if metadata and "version" in metadata:
            version = metadata["version"]
        
        # Format response
        processing_time = time.time() - start_time
        response = {
            "model_name": model_name,
            "model_version": version,
            "prediction": result,
            "processing_time_seconds": processing_time
        }
        
        # Include metadata if available
        if metadata:
            response["metadata"] = metadata
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error making prediction with latest version of {model_name}: {e}")
        return jsonify({"error": str(e)}), 500

@inference_bp.route('/models/uuid/<uuid>/predict', methods=['POST'])
def predict_by_uuid(uuid: str):
    """
    Make a prediction using a model specified by UUID.
    
    Args:
        uuid: UUID of the model
        
    Returns:
        JSON response with prediction results
    """
    try:
        # Get the model file and metadata
        start_time = time.time()
        model_path, metadata = current_app.model_service.get_model_by_uuid(uuid)
        
        if not model_path:
            return jsonify({"error": f"Model with UUID {uuid} not found or could not be loaded"}), 404
        
        # Get input data from request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        input_data = request.get_json()
        
        # Run prediction
        result = run_prediction(model_path, input_data)
        
        # Get model name and version from metadata if available
        model_name = "unknown"
        version = "unknown"
        if metadata:
            if "model_name" in metadata:
                model_name = metadata["model_name"]
            if "version" in metadata:
                version = metadata["version"]
        
        # Format response
        processing_time = time.time() - start_time
        response = {
            "model_uuid": uuid,
            "model_name": model_name,
            "model_version": version,
            "prediction": result,
            "processing_time_seconds": processing_time
        }
        
        # Include metadata if available
        if metadata:
            response["metadata"] = metadata
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error making prediction with model UUID {uuid}: {e}")
        return jsonify({"error": str(e)}), 500

@inference_bp.route('/models/list', methods=['GET'])
def list_available_models():
    """
    List all available models from the model server.
    
    Returns:
        JSON response with available models
    """
    try:
        import requests
        
        # Get models from model server
        model_url = f"{current_app.model_service.model_server_url}/models"
        response = requests.get(model_url, timeout=REQUEST_TIMEOUT)
        
        if response.status_code != 200:
            return jsonify({"error": f"Error fetching models from model server: {response.text}"}), 500
        
        return jsonify(response.json())
        
    except Exception as e:
        logger.error(f"Error listing available models: {e}")
        return jsonify({"error": str(e)}), 500

@inference_bp.route('/cache/clear', methods=['POST'])
def clear_model_cache():
    """
    Clear the model cache.
    
    Returns:
        JSON response with success message
    """
    try:
        current_app.model_service.clear_cache()
        return jsonify({"message": "Model cache cleared successfully"})
        
    except Exception as e:
        logger.error(f"Error clearing model cache: {e}")
        return jsonify({"error": str(e)}), 500

def run_prediction(model_path: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference with an ONNX model.
    
    Args:
        model_path: Path to the ONNX model file
        input_data: Input data for the model
        
    Returns:
        Prediction results
    """
    try:
        # Create ONNX inference session with appropriate provider configuration
        session_options = ort.SessionOptions()
        providers = ['CPUExecutionProvider']
        
        # For newer versions of ONNX Runtime, providers parameter is a keyword arg
        session = ort.InferenceSession(model_path, providers=providers, sess_options=session_options)
    except TypeError:
        # Fallback for older versions of ONNX Runtime where providers was a positional arg
        logger.warning("Using legacy ONNX Runtime initialization pattern")
        session = ort.InferenceSession(model_path, session_options, providers)
    
    # Get input and output names
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    
    # Prepare input tensors
    input_tensors = {}
    for name in input_names:
        if name in input_data:
            # Convert input data to numpy array with explicit float32 type
            # This ensures compatibility with models expecting float32 tensors
            tensor = np.array(input_data[name], dtype=np.float32)
            
            # Get expected shape from input definition
            input_shape = None
            for model_input in session.get_inputs():
                if model_input.name == name:
                    input_shape = model_input.shape
                    break
            
            # Handle reshaping if needed (e.g., if model expects 2D but input is 1D)
            if input_shape and len(input_shape) == 2:
                # For 2D inputs like [batch_size, features], ensure input has the right shape
                if len(tensor.shape) == 1:
                    # Convert [x] to [[x]] - reshape 1D to 2D (with batch size 1)
                    tensor = tensor.reshape(1, -1)
            
            input_tensors[name] = tensor
        else:
            raise ValueError(f"Input {name} not found in request data")
    
    # Run inference
    outputs = session.run(output_names, input_tensors)
    
    # Format results
    result = {}
    for i, name in enumerate(output_names):
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(outputs[i], np.ndarray):
            result[name] = outputs[i].tolist()
        else:
            result[name] = outputs[i]
    
    return result
