#!/usr/bin/env python3
"""
Model validation utilities.

This module provides functions for validating ONNX models before storage.
"""

import os
import logging
import numpy as np
import onnxruntime as ort
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)

class ModelValidationError(Exception):
    """Exception raised when model validation fails."""
    pass

def create_test_input(input_shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Create test input data for model validation.
    
    Args:
        input_shape: Shape of the input tensor
        dtype: Data type of the input tensor
        
    Returns:
        Numpy array with random data matching the specified shape and type
    """
    return np.random.rand(*input_shape).astype(dtype)

def validate_onnx_model(model_path: str, 
                        test_input: Optional[np.ndarray] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate an ONNX model by running a test inference.
    
    Args:
        model_path: Path to the ONNX model file
        test_input: Optional test input data. If None, a default test input will be created.
        
    Returns:
        Tuple of (is_valid, validation_info)
    """
    try:
        # Load the model
        logger.info(f"Validating ONNX model: {model_path}")
        session = ort.InferenceSession(model_path)
        
        # Get input details
        inputs = session.get_inputs()
        if not inputs:
            raise ModelValidationError("Model has no inputs")
        
        input_name = inputs[0].name
        
        # If no test input provided, create one based on the model's input shape
        if test_input is None:
            # Default to a batch of 1 with 2 features if shape info is not available
            shape = inputs[0].shape
            if shape[0] == 'batch_size' or shape[0] is None:  # Handle dynamic batch dimension
                shape = (1,) + tuple(dim if dim is not None else 2 for dim in shape[1:])
            else:
                shape = tuple(dim if dim is not None else 2 for dim in shape)
                
            test_input = create_test_input(shape)
            logger.info(f"Created test input with shape {shape}")
        
        # Run inference
        outputs = session.run(None, {input_name: test_input})
        
        # Extract model metadata
        model_info = {
            "input_name": input_name,
            "input_shape": inputs[0].shape,
            "input_type": inputs[0].type,
            "output_names": [output.name for output in session.get_outputs()],
            "output_shapes": [output.shape for output in session.get_outputs()],
            "output_types": [output.type for output in session.get_outputs()],
            "providers": session.get_providers(),
            "test_output_shape": [output.shape for output in outputs],
        }
        
        logger.info(f"Model validation successful for {model_path}")
        return True, model_info
        
    except Exception as e:
        logger.error(f"Model validation failed for {model_path}: {str(e)}")
        return False, {"error": str(e)}

def extract_model_properties(model_path: str) -> Dict[str, str]:
    """
    Extract model properties from an ONNX model.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        Dictionary of model properties
    """
    try:
        session = ort.InferenceSession(model_path)
        metadata = session.get_modelmeta()
        
        properties = {}
        if metadata.custom_metadata_map:
            properties = dict(metadata.custom_metadata_map)
            
        return properties
    except Exception as e:
        logger.error(f"Failed to extract model properties: {e}")
        return {}

def get_model_summary(model_path: str) -> Dict[str, Any]:
    """
    Get a comprehensive summary of an ONNX model.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        Dictionary with model summary information
    """
    is_valid, validation_info = validate_onnx_model(model_path)
    properties = extract_model_properties(model_path)
    
    summary = {
        "path": model_path,
        "size_bytes": os.path.getsize(model_path),
        "is_valid": is_valid,
        "validation_info": validation_info,
        "properties": properties
    }
    
    return summary
