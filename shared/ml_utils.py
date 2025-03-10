#!/usr/bin/env python3
"""
Enhanced shared ML utilities for model creation, training, export, and analysis.

This module provides comprehensive utilities for:
1. Model definition with proper normalization
2. Model training with standardized parameters
3. ONNX export with consistent settings
4. Model metadata generation
5. Model inspection and validation
6. Input/output formatting helpers

These utilities are designed to be used by all system components to ensure
consistency and reduce code duplication.
"""

import json
import os
import io
from typing import Dict, List, Any, Optional, Tuple, Union, BinaryIO
import numpy as np
import torch
import onnx
import onnxruntime as ort
from datetime import datetime

class LinearRegressionModel(torch.nn.Module):
    """
    Linear regression model with batch normalization.
    
    This model is designed to predict min_prb_ratio based on input features
    like CQI and throughput. It includes batch normalization for inputs and
    stores normalization parameters for outputs to ensure consistent predictions.
    """
    def __init__(self, input_features: int = 2, output_features: int = 1):
        """
        Initialize the model with configurable feature dimensions.
        
        Args:
            input_features: Number of input features (default: 2)
            output_features: Number of output features (default: 1)
        """
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_features, output_features)
        
        # Apply batch normalization to input features
        self.batch_norm = torch.nn.BatchNorm1d(input_features)
        
        # Register buffers to store the mean and standard deviation of the output features
        self.register_buffer('y_mean', torch.zeros(output_features))
        self.register_buffer('y_std', torch.ones(output_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor with shape [batch_size, input_features]
            
        Returns:
            Output tensor with shape [batch_size, output_features]
        """
        x_normalized = self.batch_norm(x)
        output = self.linear(x_normalized)
        
        # Denormalize output during inference
        if not self.training:
            with torch.no_grad():
                output = output * self.y_std + self.y_mean
                
        return output
    
    def get_input_shape(self) -> List[int]:
        """
        Get the expected input shape for this model.
        
        Returns:
            List representing the input shape [batch_size, input_features]
        """
        return [None, self.batch_norm.num_features]
    
    def get_output_shape(self) -> List[int]:
        """
        Get the expected output shape for this model.
        
        Returns:
            List representing the output shape [batch_size, output_features]
        """
        return [None, self.linear.out_features]


def create_and_train_model(
    input_features: int = 2,
    output_features: int = 1,
    sample_data: Optional[Dict[str, np.ndarray]] = None,
    num_epochs: int = 100,
    learning_rate: float = 0.05
) -> LinearRegressionModel:
    """
    Create and train the LinearRegressionModel with configurable parameters.
    
    Args:
        input_features: Number of input features (default: 2)
        output_features: Number of output features (default: 1)
        sample_data: Optional dict with 'features' and 'targets' numpy arrays
        num_epochs: Number of training epochs (default: 100)
        learning_rate: Learning rate for optimizer (default: 0.05)
        
    Returns:
        Trained LinearRegressionModel
    """
    # Create model with specified dimensions
    model = LinearRegressionModel(input_features, output_features)
    
    # Use provided sample data or generate default data
    if sample_data is None:
        # Default sample data (similar to what's used in playground.py)
        features = np.array([
            [10.0, 100.0],
            [8.0, 80.0],
            [6.0, 60.0],
            [4.0, 40.0],
            [2.0, 20.0]
        ], dtype=np.float32)
        
        targets = np.array([
            [50.0],
            [60.0],
            [70.0],
            [80.0],
            [90.0]
        ], dtype=np.float32)
    else:
        features = sample_data['features']
        targets = sample_data['targets']
    
    # Convert to PyTorch tensors
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    
    # Set the y_mean and y_std buffers
    model.y_mean = y.mean(dim=0, keepdim=True)
    model.y_std = y.std(dim=0, keepdim=True)
    
    # Training parameters
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Train the model
    model.train()
    for epoch in range(num_epochs):
        # Forward pass
        y_predicted = model(X)
        loss = criterion(y_predicted, (y - model.y_mean) / model.y_std)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Set to evaluation mode
    model.eval()
    return model


def export_model_to_onnx(
    model: torch.nn.Module, 
    file_path: str, 
    input_shape: Optional[List[int]] = None,
    input_names: List[str] = ["input"], 
    output_names: List[str] = ["output"],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
) -> str:
    """
    Export PyTorch model to ONNX format with configurable parameters.
    
    Args:
        model: PyTorch model to export
        file_path: Path where the ONNX model will be saved
        input_shape: Shape of the dummy input (default: [1, 2])
        input_names: Names for the input tensors (default: ["input"])
        output_names: Names for the output tensors (default: ["output"])
        dynamic_axes: Dictionary specifying dynamic axes (default: batch_size is dynamic)
        
    Returns:
        Path to the exported ONNX model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Use default input shape if not provided
    if input_shape is None:
        if hasattr(model, 'get_input_shape'):
            # Try to get shape from model if it has the method
            shape = model.get_input_shape()
            # Replace None with 1 for batch dimension
            input_shape = [1 if dim is None else dim for dim in shape]
        else:
            # Default to [1, 2] for LinearRegressionModel
            input_shape = [1, 2]
    
    # Create dummy input with the specified shape
    dummy_input = torch.randn(*input_shape)
    
    # Set up dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"}, 
            "output": {0: "batch_size"}
        }
    
    # Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        file_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    # Verify the exported model
    verify_onnx_model(file_path)
    
    return file_path


def export_model_to_buffer(
    model: torch.nn.Module,
    input_shape: Optional[List[int]] = None,
    input_names: List[str] = ["input"],
    output_names: List[str] = ["output"],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
) -> bytes:
    """
    Export PyTorch model to ONNX format in a bytes buffer.
    
    Args:
        model: PyTorch model to export
        input_shape: Shape of the dummy input (default: [1, 2])
        input_names: Names for the input tensors (default: ["input"])
        output_names: Names for the output tensors (default: ["output"])
        dynamic_axes: Dictionary specifying dynamic axes (default: batch_size is dynamic)
        
    Returns:
        Bytes containing the ONNX model
    """
    # Use default input shape if not provided
    if input_shape is None:
        if hasattr(model, 'get_input_shape'):
            # Try to get shape from model if it has the method
            shape = model.get_input_shape()
            # Replace None with 1 for batch dimension
            input_shape = [1 if dim is None else dim for dim in shape]
        else:
            # Default to [1, 2] for LinearRegressionModel
            input_shape = [1, 2]
    
    # Create dummy input with the specified shape
    dummy_input = torch.randn(*input_shape)
    
    # Set up dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"}, 
            "output": {0: "batch_size"}
        }
    
    # Create a BytesIO buffer to hold the model
    buffer = io.BytesIO()
    
    # Export the model to the buffer
    torch.onnx.export(
        model,
        dummy_input,
        buffer,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    # Reset buffer position to beginning
    buffer.seek(0)
    
    # Return the buffer contents as bytes
    return buffer.getvalue()


def verify_onnx_model(model_path_or_buffer: Union[str, bytes, BinaryIO]) -> bool:
    """
    Verify that an ONNX model is valid and can be loaded by ONNX Runtime.
    
    Args:
        model_path_or_buffer: Path to ONNX model file or bytes/buffer containing model
        
    Returns:
        True if model is valid, False otherwise
    """
    try:
        # Load and check the model with ONNX
        if isinstance(model_path_or_buffer, str):
            onnx_model = onnx.load(model_path_or_buffer)
        else:
            onnx_model = onnx.load_model_from_string(
                model_path_or_buffer if isinstance(model_path_or_buffer, bytes) 
                else model_path_or_buffer.read()
            )
        
        onnx.checker.check_model(onnx_model)
        
        # Try to create an inference session
        if isinstance(model_path_or_buffer, str):
            # Create session from file
            session = create_onnx_session(model_path_or_buffer)
        else:
            # Create session from buffer
            session = create_onnx_session_from_buffer(
                model_path_or_buffer if isinstance(model_path_or_buffer, bytes)
                else model_path_or_buffer.read()
            )
        
        return True
    except Exception as e:
        print(f"Model verification failed: {e}")
        return False


def create_onnx_session(model_path: str) -> ort.InferenceSession:
    """
    Create an ONNX inference session with version compatibility handling.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        ONNX runtime inference session
    """
    session_options = ort.SessionOptions()
    providers = ['CPUExecutionProvider']
    
    try:
        # For newer versions of ONNX Runtime, providers parameter is a keyword arg
        return ort.InferenceSession(model_path, providers=providers, sess_options=session_options)
    except TypeError:
        # Fallback for older versions of ONNX Runtime where providers was a positional arg
        return ort.InferenceSession(model_path, session_options, providers)


def create_onnx_session_from_buffer(model_buffer: bytes) -> ort.InferenceSession:
    """
    Create an ONNX inference session from a buffer with version compatibility handling.
    
    Args:
        model_buffer: Bytes containing the ONNX model
        
    Returns:
        ONNX runtime inference session
    """
    session_options = ort.SessionOptions()
    providers = ['CPUExecutionProvider']
    
    try:
        # For newer versions of ONNX Runtime, providers parameter is a keyword arg
        return ort.InferenceSession(model_buffer, providers=providers, sess_options=session_options)
    except TypeError:
        # Fallback for older versions of ONNX Runtime where providers was a positional arg
        return ort.InferenceSession(model_buffer, session_options, providers)


def get_model_input_shape(model_path_or_buffer: Union[str, bytes, BinaryIO]) -> List[Optional[int]]:
    """
    Get the expected input shape from an ONNX model.
    
    Args:
        model_path_or_buffer: Path to ONNX model file or bytes/buffer containing model
        
    Returns:
        List representing the input shape with None for dynamic dimensions
    """
    try:
        # Create session based on input type
        if isinstance(model_path_or_buffer, str):
            session = create_onnx_session(model_path_or_buffer)
        else:
            buffer = model_path_or_buffer if isinstance(model_path_or_buffer, bytes) else model_path_or_buffer.read()
            session = create_onnx_session_from_buffer(buffer)
        
        # Get input shape from the first input
        if session.get_inputs():
            input_shape = session.get_inputs()[0].shape
            
            # Convert shape to list and replace any 'None' or dynamic dimensions with None
            return [None if isinstance(dim, str) or dim is None or dim <= 0 else dim for dim in input_shape]
        
        return [None, None]  # Default shape with unknown dimensions
    except Exception as e:
        print(f"Error getting model input shape: {e}")
        return [None, None]  # Default shape with unknown dimensions


def format_input_tensor(tensor: np.ndarray, input_shape: Optional[List[Optional[int]]]) -> np.ndarray:
    """
    Format input tensor to match expected model input shape.
    
    Args:
        tensor: Input tensor as numpy array
        input_shape: Expected input shape from model
        
    Returns:
        Properly formatted tensor
    """
    # If no shape provided or shape is all None, do minimal formatting
    if input_shape is None or all(dim is None for dim in input_shape):
        # Handle reshaping if needed (e.g., if tensor is 1D, make it 2D)
        if len(tensor.shape) == 1:
            return tensor.reshape(1, -1)
        return tensor
    
    # For 2D inputs like [batch_size, features], ensure input has the right shape
    if len(input_shape) == 2 and len(tensor.shape) == 1:
        # Convert [x] to [[x]] - reshape 1D to 2D (with batch size 1)
        return tensor.reshape(1, -1)
    
    # Add batch dimension if needed
    if len(input_shape) > len(tensor.shape):
        # If tensor dimensions are fewer than model expects, add batch dimension
        return tensor.reshape(1, *tensor.shape)
    
    return tensor


def get_default_metadata(
    model_name: Optional[str] = None, 
    version: Optional[str] = None, 
    description: Optional[str] = None,
    input_features: Optional[List[str]] = None,
    output_features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create default metadata for a model with configurable fields.
    
    Args:
        model_name: Optional name of the model
        version: Optional version string
        description: Optional description of the model
        input_features: Optional list of input feature names
        output_features: Optional list of output feature names
        
    Returns:
        Dictionary with model metadata
    """
    # Default input and output features if not provided
    if input_features is None:
        input_features = ["CQI", "DRB.UEThpDl"]
    
    if output_features is None:
        output_features = ["min_prb_ratio"]
    
    metadata = {
        'description': description or 'Linear regression model for PRB prediction based on CQI and throughput',
        'training_date': datetime.now().isoformat(),
        'input_features': json.dumps(input_features),
        'output_features': json.dumps(output_features),
        'framework': f'PyTorch {torch.__version__}'
    }
    
    # Add optional fields if provided
    if model_name:
        metadata['model_name'] = model_name
    
    if version:
        metadata['version'] = version
        
    return metadata


def run_prediction(
    model_path_or_session: Union[str, bytes, ort.InferenceSession], 
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run inference with an ONNX model.
    
    Args:
        model_path_or_session: Path to the ONNX model file, bytes of model, or InferenceSession
        input_data: Input data for the model
        
    Returns:
        Prediction results
    """
    # Create session if needed
    if isinstance(model_path_or_session, str):
        session = create_onnx_session(model_path_or_session)
    elif isinstance(model_path_or_session, bytes):
        session = create_onnx_session_from_buffer(model_path_or_session)
    else:
        session = model_path_or_session
    
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
            
            # Format tensor using helper function
            tensor = format_input_tensor(tensor, input_shape)
            
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
