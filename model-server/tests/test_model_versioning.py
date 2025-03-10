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

# Add shared directory to path
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(parent_dir))  # To access shared

# Import shared ML and test utilities
from shared.ml_utils import (
    LinearRegressionModel,
    create_and_train_model,
    export_model_to_onnx,
    get_default_metadata
)

# Model server URL (default when running inside the container)
MODEL_SERVER_URL = os.environ.get('MODEL_SERVER_URL', 'http://localhost:5000')

def create_and_export_model(model_name, version):
    """Create a LinearRegressionModel and export it to ONNX format."""
    # Create and train the model using the shared utility
    model = create_and_train_model(
        input_features=2,
        output_features=1,
        num_epochs=10  # Just a few epochs for testing
    )
    
    # Create metadata with version info
    metadata = get_default_metadata(
        model_name=model_name,
        version=version,
        description="Linear regression model for PRB prediction based on CQI and throughput",
        input_features=["CQI", "DRB.UEThpDl"],
        output_features=["min_prb_ratio"]
    )
    
    # Add additional test-specific metadata
    metadata.update({
        "dataset": "network_metrics_test",
        "test_marker": f"versioning_test_{version}"
    })
    
    # Export to ONNX
    temp_dir = tempfile.mkdtemp()
    temp_model_path = os.path.join(temp_dir, f"{model_name}_v{version}.onnx")
    
    export_model_to_onnx(
        model, 
        temp_model_path, 
        input_names=["input"], 
        output_names=["output"], 
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    # Add metadata after export using onnx library (optional since we'll send it separately)
    onnx_model = onnx.load(temp_model_path)
    
    # Add metadata as model properties
    for key, value in metadata.items():
        meta = onnx_model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    
    # Save the model with metadata
    onnx.save(onnx_model, temp_model_path)
    
    return temp_dir, temp_model_path, metadata

def upload_model(model_name, version, model_path, metadata):
    """Upload a model to the model server."""
    with open(model_path, 'rb') as f:
        model_data = f.read()
        
    # Create a named file tuple to ensure it uses the correct filename
    filename = f"{model_name}_v{version}.onnx"
    files = {'model': (filename, model_data)}
    form_data = {'metadata': json.dumps(metadata)}
    
    # First, delete any existing model with this name and version to avoid conflicts
    try:
        delete_response = requests.delete(f"{MODEL_SERVER_URL}/models/{model_name}/versions/{version}")
    except Exception:
        pass
        
    # Now upload the model
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
        except Exception:
            pass

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
    
    # Verify models were added to the server
    list_response = requests.get(f"{MODEL_SERVER_URL}/models")
    assert list_response.status_code == 200, f"Failed to list models: {list_response.text}"
    models = list_response.json()
    assert model_name in models, f"Model '{model_name}' not found in models list: {models}"
    
    # Verify versions are available
    versions_response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions")
    assert versions_response.status_code == 200, f"Failed to get model versions: {versions_response.text}"
    
    yield model_name
    
    # Clean up temporary files
    import os
    try:
        if os.path.exists(model_path1):
            os.remove(model_path1)
        if os.path.exists(model_path2):
            os.remove(model_path2)
        if os.path.exists(temp_dir1):
            os.rmdir(temp_dir1)
        if os.path.exists(temp_dir2):
            os.rmdir(temp_dir2)
        
        # Delete test models from server
        delete_model_versions(model_name, ["1.0.0", "1.0.1"])
    except Exception:
        pass

def test_upload_models(test_models):
    """Test uploading multiple versions of a model."""
    model_name = test_models
    
    # Get initial model count
    initial_response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions")
    assert initial_response.status_code == 200
    initial_versions = initial_response.json()
    initial_count = len(initial_versions.get("versions", []))
    
    # Try to upload a new version of a model
    new_version = "1.1.0"  # Use a version that likely doesn't exist
    temp_dir, model_path, metadata = create_and_export_model(model_name, new_version)
    try:
        # Upload
        response = upload_model(model_name, new_version, model_path, metadata)
        assert response.status_code == 200, f"Failed to upload new version: {response.text}"
        
        # Verify the model count increased
        after_response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions")
        assert after_response.status_code == 200
        after_versions = after_response.json()
        after_count = len(after_versions.get("versions", []))
        
        # We should have one more version now
        assert after_count == initial_count + 1, f"Expected version count to increase by 1, was {initial_count}, now {after_count}"
        
        # The new version should be in the list
        version_found = False
        for version in after_versions.get("versions", []):
            if version["version"] == new_version:
                version_found = True
                break
        assert version_found, f"New version {new_version} not found in versions list"
    finally:
        # Clean up
        os.remove(model_path)
        os.rmdir(temp_dir)
        # Delete the new version
        requests.delete(f"{MODEL_SERVER_URL}/models/{model_name}/versions/{new_version}")

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
    
    # Verify both versions are available, regardless of ordering
    versions_found = set(v["version"] for v in versions["versions"])
    assert "1.0.0" in versions_found, "Version 1.0.0 should be available"
    assert "1.0.1" in versions_found, "Version 1.0.1 should be available"

def test_get_latest_version(test_models):
    """Test getting the latest version of a model."""
    model_name = test_models
    
    # First, verify the model exists and has versions by getting a list
    list_response = requests.get(f"{MODEL_SERVER_URL}/models")
    assert list_response.status_code == 200, f"Failed to list models: {list_response.text}"
    models = list_response.json()
    
    assert model_name in models, f"Model {model_name} not found in model list: {models}"
    assert len(models[model_name]) >= 1, f"No versions found for model {model_name}: {models}"
    
    # Explicitly recreate (delete & upload) the 1.0.1 version since it should be latest
    # This ensures we have a fresh model file in the storage
    temp_dir, model_path, metadata = create_and_export_model(model_name, "1.0.1")
    try:
        upload_response = upload_model(model_name, "1.0.1", model_path, metadata)
        assert upload_response.status_code == 200, f"Failed to upload model: {upload_response.text}"
        
        # Now get the latest version
        response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions/latest")
        assert response.status_code == 200, f"Failed to get latest version: {response.text}"
        
        # Since it's a binary file, just check that it has content
        assert len(response.content) > 0, "Model file is empty"
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

def test_get_specific_version(test_models):
    """Test getting a specific version of a model."""
    model_name = test_models
    version = "1.0.0"  # Use version 1.0.0 for this test
    
    # Explicitly recreate the 1.0.0 version to ensure we have a fresh model file
    temp_dir, model_path, metadata = create_and_export_model(model_name, version)
    try:
        upload_response = upload_model(model_name, version, model_path, metadata)
        assert upload_response.status_code == 200, f"Failed to upload model: {upload_response.text}"
        
        # List models to make sure it exists
        list_response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions")
        assert list_response.status_code == 200, f"Failed to list versions: {list_response.text}"
        
        # Now get the specific version
        response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions/{version}")
        assert response.status_code == 200, f"Failed to get specific version: {response.text}"
        
        # Since it's a binary file, just check that it has content
        assert len(response.content) > 0, "Model file is empty"
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

def test_get_model_metadata(test_models):
    """Test getting metadata for a model version."""
    model_name = test_models
    version = "1.0.0"  # Use version 1.0.0 for this test
    
    # Explicitly recreate the 1.0.0 version to ensure we have fresh metadata
    temp_dir, model_path, metadata = create_and_export_model(model_name, version)
    try:
        # Include very specific metadata we can test for
        metadata["test_marker"] = "metadata_test_marker_1234"
        upload_response = upload_model(model_name, version, model_path, metadata)
        assert upload_response.status_code == 200, f"Failed to upload model with metadata: {upload_response.text}"
        
        # Now get the metadata
        response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions/{version}/metadata")
        assert response.status_code == 200, f"Failed to get metadata: {response.text}"
        
        retrieved_metadata = response.json()
        
        # Check the metadata
        assert "version" in retrieved_metadata, "Metadata missing version information"
        assert retrieved_metadata["version"] == version, f"Expected version {version}, got {retrieved_metadata.get('version')}"
        assert "description" in retrieved_metadata, "Metadata missing description"
        assert "network_metrics_test" == retrieved_metadata.get("dataset", ""), "Metadata missing or incorrect dataset info"
        assert retrieved_metadata.get("test_marker") == "metadata_test_marker_1234", "Custom metadata marker not found"
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
