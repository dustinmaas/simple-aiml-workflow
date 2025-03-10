#!/usr/bin/env python3
"""
Helper script to create test models for the inference server tests.

This script creates the same LinearRegressionModel used in the playground.py notebook
and exports it to ONNX format for testing with the inference server and model server.
"""

import os
import sys
import json
import requests
import torch
import numpy as np
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.constants import MODEL_SERVER_URL

class LinearRegressionModel(torch.nn.Module):
    """
    Linear regression model with batch normalization as used in playground.py.
    This model is designed to predict min_prb_ratio based on CQI and throughput.
    """
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)  # two input features, one output feature
        
        # Apply batch normalization to input features
        self.batch_norm = torch.nn.BatchNorm1d(2)
        
        # Register buffers to store the mean and standard deviation of the output features
        self.register_buffer('y_mean', torch.zeros(1))
        self.register_buffer('y_std', torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = self.batch_norm(x)
        output = self.linear(x_normalized)
        
        if not self.training:
            with torch.no_grad():
                output = output * self.y_std + self.y_mean
                
        return output

def create_and_train_model():
    """Create and train the LinearRegressionModel with sample data."""
    # Sample data (similar to what's used in playground.py)
    # Input features: CQI, DRB.UEThpDl
    features = np.array([
        [10.0, 100.0],
        [8.0, 80.0],
        [6.0, 60.0],
        [4.0, 40.0],
        [2.0, 20.0]
    ], dtype=np.float32)
    
    # Target: min_prb_ratio
    targets = np.array([
        [50.0],
        [60.0],
        [70.0],
        [80.0],
        [90.0]
    ], dtype=np.float32)
    
    # Convert to PyTorch tensors
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    
    # Create model
    model = LinearRegressionModel()
    
    # Set the y_mean and y_std buffers
    model.y_mean = y.mean(dim=0, keepdim=True)
    model.y_std = y.std(dim=0, keepdim=True)
    
    # Training parameters
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    
    # Train the model
    model.train()
    num_epochs = 100
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

def create_test_model():
    """Create the LinearRegressionModel and upload it to the model server."""
    
    # Create and train the model
    model = create_and_train_model()
    
    # PyTorch needs a dummy input for tracing operations
    dummy_input = torch.randn(1, 2)
    
    # Create temp file for ONNX model
    temp_model_path = "/tmp/test_inference_model.onnx"
    
    # Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        temp_model_path,
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    # Upload model to model server
    model_name = "test_inference_model"
    version = "1.0.0"
    
    with open(temp_model_path, 'rb') as f:
        metadata = {
            'model_name': model_name,
            'version': version,
            'description': 'Linear regression model for PRB prediction based on CQI and throughput',
            'training_date': datetime.now().isoformat(),
            'input_features': json.dumps(["CQI", "DRB.UEThpDl"]),
            'output_features': json.dumps(["min_prb_ratio"]),
            'framework': f'PyTorch {torch.__version__}'
        }
        
        upload_response = requests.post(
            f"{MODEL_SERVER_URL}/models/{model_name}/versions/{version}",
            files={'model': f},
            data=metadata
        )
        
        if upload_response.status_code == 200:
            model_uuid = upload_response.json().get('uuid')
            print(f"Successfully uploaded test model. UUID: {model_uuid}")
            return {
                "model_name": model_name,
                "version": version,
                "uuid": model_uuid
            }
        else:
            print(f"Failed to upload test model: {upload_response.text}")
            return None

def create_versioned_test_models():
    """Create multiple versions of the LinearRegressionModel for versioning tests."""
    
    model_name = "test_versioning_model"
    versions = ["1.0.0", "1.1.0"]
    results = []
    
    # Create and train base model
    model = create_and_train_model()
    dummy_input = torch.randn(1, 2)
    
    for version in versions:
        # Export model to temp file with version in filename
        temp_model_path = f"/tmp/test_model_{version}.onnx"
        
        # Make slight parameter adjustments for different versions
        if version != "1.0.0":
            # Slightly modify weights for different versions
            with torch.no_grad():
                model.linear.weight.data = model.linear.weight.data * 1.05
                model.linear.bias.data = model.linear.bias.data + 0.1
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            temp_model_path,
            verbose=False,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        
        # Upload to model server
        with open(temp_model_path, 'rb') as f:
            metadata = {
                'model_name': model_name,
                'version': version,
                'description': f'Linear regression model version {version} for PRB prediction',
                'training_date': datetime.now().isoformat(),
                'input_features': json.dumps(["CQI", "DRB.UEThpDl"]),
                'output_features': json.dumps(["min_prb_ratio"]),
                'framework': f'PyTorch {torch.__version__}'
            }
            
            upload_response = requests.post(
                f"{MODEL_SERVER_URL}/models/{model_name}/versions/{version}",
                files={'model': f},
                data=metadata
            )
            
            if upload_response.status_code == 200:
                model_uuid = upload_response.json().get('uuid')
                print(f"Successfully uploaded {model_name} version {version}. UUID: {model_uuid}")
                results.append({
                    "model_name": model_name,
                    "version": version,
                    "uuid": model_uuid
                })
            else:
                print(f"Failed to upload {model_name} version {version}: {upload_response.text}")
                
    return results

if __name__ == "__main__":
    # Check if model server is available
    try:
        response = requests.get(f"{MODEL_SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            # Create a simple test model
            result = create_test_model()
            if result:
                print(f"Test model created: {result}")
            else:
                print("Failed to create test model")
                
            # Create versioned models for versioning tests
            versioned_models = create_versioned_test_models()
            if versioned_models:
                print(f"Created {len(versioned_models)} versioned test models")
            else:
                print("Failed to create versioned test models")
        else:
            print(f"Model server health check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to model server: {e}")
