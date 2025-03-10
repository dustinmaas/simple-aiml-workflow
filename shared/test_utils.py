#!/usr/bin/env python3
"""
Shared test utilities for model-server and inference-server tests.

This module provides common utilities for testing including:
- Model download and analysis functions
- Input data formatting based on model shape
- Model server connectivity checks
- Model and metadata retrieval
"""

import os
import onnx
import numpy as np
import requests
from typing import List, Dict, Any, Optional, Tuple, Union

def download_and_analyze_model(model_url: str, temp_path: str = "/tmp/temp_model.onnx") -> Tuple[List[Optional[int]], Dict[str, Any]]:
    """
    Download a model from the given URL and analyze its input shape.
    
    Args:
        model_url: URL to download the model from
        temp_path: Path to save the temporary model file
        
    Returns:
        Tuple of (input_shape, model_metadata)
    """
    try:
        model_response = requests.get(model_url, timeout=10)
        if model_response.status_code != 200:
            return [None, None], {}
            
        # Save the model to a temporary file
        with open(temp_path, "wb") as f:
            f.write(model_response.content)
        
        # Load and analyze the model
        onnx_model = onnx.load(temp_path)
        input_tensor = onnx_model.graph.input[0]
        input_shape = []
        
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.dim_value:
                input_shape.append(dim.dim_value)
            else:
                input_shape.append(None)  # Dynamic dimension
                
        return input_shape, {}
        
    except Exception as e:
        print(f"Error analyzing model: {e}")
        return [None, None], {}

def create_input_data_for_shape(shape: List[Optional[int]], test_values: List[float] = [10.0, 100.0]) -> Dict[str, Any]:
    """
    Create properly formatted input data based on the model's expected shape.
    
    Args:
        shape: The input shape from the model
        test_values: Test values to use for inputs
        
    Returns:
        Dictionary with properly formatted input data
    """
    if not shape or len(shape) < 2:
        # Fallback: Use batched format with 2 features
        return {"input": [[10.0, 100.0]]}
        
    if shape[0] is None:  # First dimension is batch size (typically None/dynamic)
        if shape[1] == 1:
            # Model expects a column vector [batch_size, 1]
            return {"input": [[v] for v in test_values]}
        elif shape[1] == 2:
            # Model expects [batch_size, 2]
            return {"input": [test_values]}
        else:
            # Unknown feature count, use our test values
            return {"input": [test_values]}
    else:
        # Fixed batch size
        if shape[1] == 1:
            # Model expects a column vector with fixed batch size
            return {"input": [[v] for v in test_values[:shape[0]]]}
        else:
            # Model expects a fixed batch size with multiple features
            return {"input": [test_values[:shape[1]]]}

def check_model_server_availability(model_server_url: str) -> bool:
    """
    Check if the model server is available.
    
    Args:
        model_server_url: URL of the model server
        
    Returns:
        True if available, False otherwise
    """
    try:
        response = requests.get(f"{model_server_url}/health", timeout=5)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False

def get_available_models(model_server_url: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get available models from the model server.
    
    Args:
        model_server_url: URL of the model server
        
    Returns:
        Dictionary of models or empty dict if no models available
    """
    try:
        response = requests.get(f"{model_server_url}/models", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception:
        return {}

def get_model_detail(model_server_url: str, model_uuid: str) -> Optional[Dict[str, Any]]:
    """
    Get model details by UUID.
    
    Args:
        model_server_url: URL of the model server
        model_uuid: UUID of the model
        
    Returns:
        Model detail dictionary or None if not found
    """
    try:
        response = requests.get(f"{model_server_url}/models/uuid/{model_uuid}/detail", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

def get_model_metadata(model_server_url: str, model_uuid: str) -> Optional[Dict[str, Any]]:
    """
    Get model metadata by UUID.
    
    Args:
        model_server_url: URL of the model server
        model_uuid: UUID of the model
        
    Returns:
        Model metadata dictionary or None if not found
    """
    try:
        response = requests.get(f"{model_server_url}/models/uuid/{model_uuid}/metadata", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

def extract_storage_uuid(file_path: str) -> Optional[str]:
    """
    Extract the storage UUID from a file path.
    
    Args:
        file_path: Path to the model file
        
    Returns:
        Extracted UUID or None if not found
    """
    if not file_path:
        return None
    
    # Extract basename and remove extension
    return os.path.basename(file_path).replace(".onnx", "")
