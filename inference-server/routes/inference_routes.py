#!/usr/bin/env python3
"""
Enhanced inference routes for the inference server.

This module defines routes for model inference operations including:
- Getting predictions from models by UUID, name/version, or latest version
- Model metadata retrieval
- Model cache management

Includes improved error handling, request validation, and uses enhanced shared utilities.
"""

import os
import logging
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Union
import onnxruntime as ort
from flask import Blueprint, request, jsonify, current_app, Response
import numpy as np

from utils.constants import REQUEST_TIMEOUT
from shared.ml_utils import (
    create_onnx_session, 
    format_input_tensor, 
    run_prediction,
    get_model_input_shape
)

logger = logging.getLogger(__name__)

# Create Blueprint for inference routes
inference_bp = Blueprint('inference', __name__, url_prefix='/inference')

class InferenceError(Exception):
    """Base class for inference-related errors."""
    pass

class ModelLoadError(InferenceError):
    """Error raised when a model cannot be loaded."""
    pass

class InvalidInputError(InferenceError):
    """Error raised when input data is invalid."""
    pass

def validate_input_data() -> Dict[str, Any]:
    """
    Validate and extract input data from the request.
    
    Returns:
        Dictionary with validated input data
        
    Raises:
        InvalidInputError: If request is not JSON or has invalid structure
    """
    if not request.is_json:
        raise InvalidInputError("Request must be JSON")
    
    input_data = request.get_json()
    
    # Additional validation could be added here
    # For example, checking for required fields or data types
    
    return input_data

def _handle_prediction_request(
    model_path: Optional[str], 
    metadata: Optional[Dict[str, Any]], 
    model_identifier: Dict[str, str]
) -> Tuple[Response, int]:
    """
    Handle common prediction logic for all prediction endpoints.
    
    Args:
        model_path: Path to the model file (None if model not found)
        metadata: Model metadata (None if not available)
        model_identifier: Dictionary with model identification (name, version, uuid)
        
    Returns:
        Tuple of (response, status_code)
        
    Raises:
        ModelLoadError: If model cannot be loaded
        InvalidInputError: If input data is invalid
    """
    try:
        if not model_path:
            raise ModelLoadError("Model not found or could not be loaded")
        
        # Validate input data
        input_data = validate_input_data()
        
        # Measure processing time
        start_time = time.time()
        
        # Run prediction using shared utility
        result = run_prediction(model_path, input_data)
        
        # Format response
        processing_time = time.time() - start_time
        response = {
            **model_identifier,
            "prediction": result,
            "processing_time_seconds": processing_time
        }
        
        # Include metadata if available
        if metadata:
            response["metadata"] = metadata
        
        return jsonify(response), 200
        
    except ModelLoadError as e:
        logger.error(f"Model load error: {e}")
        return jsonify({"error": str(e)}), 404
    except InvalidInputError as e:
        logger.error(f"Invalid input error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": f"Error processing prediction: {str(e)}"}), 500

@inference_bp.route('/models/<model_name>/versions/<version>/predict', methods=['POST'])
def predict_by_name_version(model_name: str, version: str) -> Tuple[Response, int]:
    """
    Make a prediction using a specific model version.
    
    Args:
        model_name: Name of the model
        version: Specific version to use
        
    Returns:
        JSON response with prediction results
    """
    # Get the model file and metadata
    model_path, metadata = current_app.model_service.get_model_by_name_version(model_name, version)
    
    # Process prediction using common handler
    return _handle_prediction_request(
        model_path=model_path,
        metadata=metadata,
        model_identifier={"model_name": model_name, "model_version": version}
    )

@inference_bp.route('/models/<model_name>/latest/predict', methods=['POST'])
def predict_latest_version(model_name: str) -> Tuple[Response, int]:
    """
    Make a prediction using the latest version of a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        JSON response with prediction results
    """
    # Get the latest model version
    model_path, metadata = current_app.model_service.get_latest_model_version(model_name)
    
    # Extract version from metadata or use default
    version = metadata.get("version", "unknown") if metadata else "unknown"
    
    # Process prediction using common handler
    return _handle_prediction_request(
        model_path=model_path,
        metadata=metadata,
        model_identifier={"model_name": model_name, "model_version": version}
    )

@inference_bp.route('/models/uuid/<uuid>/predict', methods=['POST'])
def predict_by_uuid(uuid: str) -> Tuple[Response, int]:
    """
    Make a prediction using a model specified by UUID.
    
    Args:
        uuid: UUID of the model
        
    Returns:
        JSON response with prediction results
    """
    # Get the model file and metadata
    model_path, metadata = current_app.model_service.get_model_by_uuid(uuid)
    
    # Extract model name and version from metadata or use defaults
    model_name = metadata.get("model_name", "unknown") if metadata else "unknown"
    version = metadata.get("version", "unknown") if metadata else "unknown"
    
    # Process prediction using common handler
    return _handle_prediction_request(
        model_path=model_path,
        metadata=metadata,
        model_identifier={"model_uuid": uuid, "model_name": model_name, "model_version": version}
    )

@inference_bp.route('/models/list', methods=['GET'])
def list_available_models() -> Tuple[Response, int]:
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
            return jsonify({
                "error": f"Error fetching models from model server: {response.text}"
            }), response.status_code
        
        return jsonify(response.json()), 200
        
    except requests.exceptions.Timeout:
        error_msg = f"Timeout connecting to model server (timeout: {REQUEST_TIMEOUT}s)"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 504
    except requests.exceptions.ConnectionError:
        error_msg = "Connection error to model server"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 502
    except Exception as e:
        logger.error(f"Error listing available models: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@inference_bp.route('/cache/clear', methods=['POST'])
def clear_model_cache() -> Tuple[Response, int]:
    """
    Clear the model cache.
    
    Returns:
        JSON response with success message
    """
    try:
        current_app.model_service.clear_cache()
        return jsonify({"message": "Model cache cleared successfully"}), 200
        
    except Exception as e:
        logger.error(f"Error clearing model cache: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@inference_bp.route('/models/<model_name>/versions/<version>/metadata', methods=['GET'])
def get_model_metadata(model_name: str, version: str) -> Tuple[Response, int]:
    """
    Get metadata for a specific model version.
    
    Args:
        model_name: Name of the model
        version: Specific version
        
    Returns:
        JSON response with model metadata
    """
    try:
        _, metadata = current_app.model_service.get_model_by_name_version(model_name, version)
        
        if not metadata:
            return jsonify({"error": "Metadata not found"}), 404
            
        return jsonify(metadata), 200
        
    except Exception as e:
        logger.error(f"Error retrieving metadata for {model_name} version {version}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@inference_bp.route('/models/uuid/<uuid>/metadata', methods=['GET'])
def get_model_metadata_by_uuid(uuid: str) -> Tuple[Response, int]:
    """
    Get metadata for a model by UUID.
    
    Args:
        uuid: UUID of the model
        
    Returns:
        JSON response with model metadata
    """
    try:
        _, metadata = current_app.model_service.get_model_by_uuid(uuid)
        
        if not metadata:
            return jsonify({"error": "Metadata not found"}), 404
            
        return jsonify(metadata), 200
        
    except Exception as e:
        logger.error(f"Error retrieving metadata for model UUID {uuid}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@inference_bp.route('/models/<model_name>/latest/metadata', methods=['GET'])
def get_latest_model_metadata(model_name: str) -> Tuple[Response, int]:
    """
    Get metadata for the latest version of a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        JSON response with model metadata
    """
    try:
        _, metadata = current_app.model_service.get_latest_model_version(model_name)
        
        if not metadata:
            return jsonify({"error": "Metadata not found"}), 404
            
        return jsonify(metadata), 200
        
    except Exception as e:
        logger.error(f"Error retrieving metadata for latest version of {model_name}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@inference_bp.route('/models/info', methods=['GET'])
def get_model_info() -> Tuple[Response, int]:
    """
    Get information about the inference server and loaded models.
    
    Returns:
        JSON response with server information
    """
    try:
        # Get server information
        info = {
            "server_version": "1.0.0",  # This could be fetched from a version file
            "onnx_runtime_version": ort.__version__,
            "loaded_model_count": current_app.model_service.get_cache_info().get("count", 0),
            "available_providers": ort.get_available_providers()
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"Error getting server info: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
