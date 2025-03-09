#!/usr/bin/env python3
"""
Pytest-based tests for the model versioning functionality in the model server.

This module tests:
1. Creating and uploading models with version information
2. Listing available model versions
3. Retrieving specific model versions
4. Getting the latest version of a model
5. Getting model metadata
"""

import os
import torch
import torch.onnx
import json
import requests
import tempfile
import pytest
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

def create_and_export_model(model_name, version):
    """Create a simple test model and export it to ONNX format."""
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
    onnx_model = onnx.load(temp_model_path)
    
    # Add metadata as model properties
    for key, value in metadata_props.items():
        meta = onnx_model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    
    # Save the model with metadata
    onnx.save(onnx_model, temp_model_path)
    
    return temp_dir, temp_model_path, metadata_props

def upload_model(model_name, version, model_path, metadata):
    """Upload a model to the model server."""
    with open(model_path, 'rb') as f:
        files = {'model': f}
        form_data = {'metadata': json.dumps(metadata)}
        response = requests.post(
            f"{MODEL_SERVER_URL}/models/{model_name}/versions/{version}",
            files=files,
            data=form_data
        )
    
    return response

def delete_model_versions(model_name, versions):
    """Delete specific model versions if they exist."""
    for version in versions:
        try:
            requests.delete(f"{MODEL_SERVER_URL}/models/{model_name}/versions/{version}")
        except Exception as e:
            print(f"Error deleting model {model_name} version {version}: {e}")

@pytest.fixture(scope="module")
def test_models():
    """Fixture to create and upload test models."""
    model_name = "test_model"
    
    # Delete any existing test models first
    delete_model_versions(model_name, ["1.0.0", "1.0.1"])
    
    # Create and export models
    temp_dir1, model_path1, metadata1 = create_and_export_model(model_name, "1.0.0")
    temp_dir2, model_path2, metadata2 = create_and_export_model(model_name, "1.0.1")
    
    # Upload models
    response1 = upload_model(model_name, "1.0.0", model_path1, metadata1)
    response2 = upload_model(model_name, "1.0.1", model_path2, metadata2)
    
    # Check if uploads were successful
    assert response1.status_code == 200, f"Failed to upload model 1.0.0: {response1.text}"
    assert response2.status_code == 200, f"Failed to upload model 1.0.1: {response2.text}"
    
    yield model_name
    
    # Clean up temporary files
    import os
    os.remove(model_path1)
    os.remove(model_path2)
    os.rmdir(temp_dir1)
    os.rmdir(temp_dir2)

def test_upload_models(test_models):
    """Test uploading multiple versions of a model."""
    model_name = test_models
    
    # Try to upload a model with an existing version (should fail with 409)
    temp_dir, model_path, metadata = create_and_export_model(model_name, "1.0.0")
    response = upload_model(model_name, "1.0.0", model_path, metadata)
    
    # Clean up
    os.remove(model_path)
    os.rmdir(temp_dir)
    
    # This should fail with 409 Conflict since we're trying to upload a duplicate version
    assert response.status_code == 409, "Expected 409 Conflict for duplicate version upload"
    assert "already exists" in response.text, "Expected 'already exists' message in error response"

def test_list_models(test_models):
    """Test listing all models."""
    model_name = test_models
    
    response = requests.get(f"{MODEL_SERVER_URL}/models")
    assert response.status_code == 200, f"Failed to list models: {response.text}"
    
    models = response.json()
    assert model_name in models, f"Model {model_name} not found in model list"
    assert len(models[model_name]) >= 2, f"Expected at least 2 versions for {model_name}"

def test_list_model_versions(test_models):
    """Test listing versions of a specific model."""
    model_name = test_models
    
    response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions")
    assert response.status_code == 200, f"Failed to list model versions: {response.text}"
    
    versions = response.json()
    assert "versions" in versions, "Response missing 'versions' key"
    assert len(versions["versions"]) >= 2, f"Expected at least 2 versions, got {len(versions.get('versions', []))}"
    
    # Check version ordering
    assert versions["versions"][0]["version"] == "1.0.0", "First version should be 1.0.0"
    assert versions["versions"][1]["version"] == "1.0.1", "Second version should be 1.0.1"

def test_get_latest_version(test_models):
    """Test getting the latest version of a model."""
    model_name = test_models
    
    response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions/latest")
    assert response.status_code == 200, f"Failed to get latest version: {response.text}"
    
    # We can't easily check the content as it's a binary file
    # But status code 200 indicates success

def test_get_specific_version(test_models):
    """Test getting a specific version of a model."""
    model_name = test_models
    
    response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions/1.0.0")
    assert response.status_code == 200, f"Failed to get specific version: {response.text}"
    
    # We can't easily check the content as it's a binary file
    # But status code 200 indicates success

def test_get_model_metadata(test_models):
    """Test getting metadata for a model version."""
    model_name = test_models
    
    response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions/1.0.0/metadata")
    assert response.status_code == 200, f"Failed to get metadata: {response.text}"
    
    metadata = response.json()
    assert "version" in metadata, "Metadata missing version information"
    assert metadata["version"] == "1.0.0", f"Expected version 1.0.0, got {metadata.get('version')}"
    assert "description" in metadata, "Metadata missing description"
    assert "test_dataset" in metadata["dataset"], "Metadata missing or incorrect dataset info"
