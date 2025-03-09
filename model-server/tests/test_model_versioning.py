#!/usr/bin/env python3
"""
Test script for the model versioning functionality in the model server.

This script tests:
1. Creating and uploading models with version information
2. Listing available model versions
3. Retrieving specific model versions
4. Getting the latest version of a model
"""

import os
import sys
import torch
import torch.onnx
import json
import requests
import tempfile
import numpy as np
from datetime import datetime
import onnx

# Model server URL (default when running inside the container)
MODEL_SERVER_URL = os.environ.get('MODEL_SERVER_URL', 'http://localhost:5000')

class SimpleLinearModel(torch.nn.Module):
    """Simple linear model for testing."""
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)
        
    def forward(self, x):
        return self.linear(x)

def create_and_upload_model(model_name, version):
    """Create a simple test model and upload it with versioning information."""
    print(f"Creating test model: {model_name} (version {version})...")
    
    # Create a simple model
    model = SimpleLinearModel()
    model.eval()
    
    # Create sample input
    dummy_input = torch.randn(1, 2)
    
    # Create metadata with version info
    metadata_props = {
        "version": version,
        "training_date": datetime.now().isoformat(),
        "framework": f"PyTorch {torch.__version__}",
        "dataset": "test_dataset",
        "metrics": json.dumps({"accuracy": 0.95}),
        "description": "Test model for versioning system",
        "input_features": json.dumps(["feature1", "feature2"]),
        "output_features": json.dumps(["prediction"])
    }
    
    # Export to ONNX
    temp_dir = tempfile.mkdtemp()
    temp_model_path = os.path.join(temp_dir, f"{model_name}_v{version}.onnx")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        temp_model_path, 
        verbose=True, 
        input_names=["input"], 
        output_names=["output"], 
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    # Add metadata after export using onnx library
    import onnx
    onnx_model = onnx.load(temp_model_path)
    
    # Add metadata as model properties
    for key, value in metadata_props.items():
        meta = onnx_model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    
    # Save the model with metadata
    onnx.save(onnx_model, temp_model_path)
    
    print(f"Model exported to {temp_model_path}")
    
    # Upload to model server with separate metadata
    print(f"Uploading model to {MODEL_SERVER_URL}/models/{model_name}/versions/{version}...")
    
    with open(temp_model_path, 'rb') as f:
        files = {'model': f}
        form_data = {'metadata': json.dumps(metadata_props)}
        response = requests.post(
            f"{MODEL_SERVER_URL}/models/{model_name}/versions/{version}",
            files=files,
            data=form_data
        )
    
    if response.status_code == 200:
        print(f"Model uploaded successfully: {response.json()}")
        success = True
    else:
        print(f"Error uploading model: {response.status_code} - {response.text}")
        success = False
    
    # Clean up
    os.remove(temp_model_path)
    os.rmdir(temp_dir)
    
    return success

def test_versioning_apis():
    """Test the versioning APIs of the model server."""
    model_name = "test_model"
    
    # Test 1: Upload two versions of the model
    print("\n--- Test 1: Uploading multiple model versions ---")
    success1 = create_and_upload_model(model_name, "1.0.0")
    success2 = create_and_upload_model(model_name, "1.0.1")
    
    if not (success1 and success2):
        print("Failed to upload test models.")
        return False
    
    # Test 2: List all models
    print("\n--- Test 2: Listing all models ---")
    response = requests.get(f"{MODEL_SERVER_URL}/models")
    if response.status_code == 200:
        models = response.json()
        print(json.dumps(models, indent=2))
        if not models or model_name not in models:
            print(f"Error: Did not find {model_name} in model list")
            return False
    else:
        print(f"Error listing models: {response.status_code} - {response.text}")
        return False
    
    # Test 3: List model versions
    print("\n--- Test 3: Listing model versions ---")
    response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions")
    if response.status_code == 200:
        versions = response.json()
        print(json.dumps(versions, indent=2))
        if not versions.get("versions") or len(versions.get("versions")) != 2:
            print(f"Error: Expected 2 versions, got {len(versions.get('versions', []))}")
            return False
    else:
        print(f"Error listing model versions: {response.status_code} - {response.text}")
        return False
    
    # Test 4: Get latest version
    print("\n--- Test 4: Getting latest version ---")
    response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions/latest")
    if response.status_code == 200:
        print(f"Successfully retrieved latest version")
    else:
        print(f"Error getting latest version: {response.status_code} - {response.text}")
        return False
    
    # Test 5: Get specific version
    print("\n--- Test 5: Getting specific version ---")
    response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions/1.0.0")
    if response.status_code == 200:
        print(f"Successfully retrieved version 1.0.0")
    else:
        print(f"Error getting specific version: {response.status_code} - {response.text}")
        return False
    
    # Test 6: Get model metadata
    print("\n--- Test 6: Getting model metadata ---")
    response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions/1.0.0/metadata")
    if response.status_code == 200:
        metadata = response.json()
        print(json.dumps(metadata, indent=2))
        if not metadata or "version" not in metadata:
            print(f"Error: Metadata missing version information")
            return False
        print(f"Successfully retrieved metadata for version 1.0.0")
    else:
        print(f"Error getting metadata: {response.status_code} - {response.text}")
        return False
    
    print("\nAll tests passed!")
    return True

def main():
    """Main function to run the tests."""
    print(f"Testing model server versioning APIs at {MODEL_SERVER_URL}")
    
    if test_versioning_apis():
        print("\nModel versioning system is working correctly!")
        return 0
    else:
        print("\nModel versioning tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
