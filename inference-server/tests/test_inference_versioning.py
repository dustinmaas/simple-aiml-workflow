#!/usr/bin/env python3
"""
Pytest-based tests for the inference versioning functionality in the inference server.

This module tests:
1. Making predictions using the latest model version
2. Making predictions using a specific model version
3. Handling model metadata in responses
"""

import os
import json
import requests
import pytest
import numpy as np
from datetime import datetime

# Server URLs (default when running inside the containers)
INFERENCE_SERVER_URL = os.environ.get('INFERENCE_SERVER_URL', 'http://localhost:5000')
MODEL_SERVER_URL = os.environ.get('MODEL_SERVER_URL', 'http://model-server:5000')

@pytest.fixture(scope="module")
def test_models():
    """Fixture to set up and verify test models on the model server."""
    # Check if we can access the model server
    try:
        response = requests.get(f"{MODEL_SERVER_URL}/health")
        assert response.status_code == 200, f"Model server health check failed: {response.status_code}"
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Error connecting to model server: {e}")
    
    # For inference testing, we'll assume the model-server tests have created the
    # necessary test models. We just need to verify they exist.
    response = requests.get(f"{MODEL_SERVER_URL}/models")
    assert response.status_code == 200, f"Failed to get models from model server: {response.text}"
    
    models = response.json()
    assert models, "No models found on the model server. Please run the model-server tests first."
    
    # Get the first model for testing
    model_name = list(models.keys())[0] if isinstance(models, dict) else "test_model"
    
    # Check if the model has versions
    response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions")
    assert response.status_code == 200, f"Failed to get model versions: {response.text}"
    
    versions_data = response.json()
    versions = versions_data.get("versions", [])
    assert versions, f"No versions found for model {model_name}"
    
    # Get the specific version for testing
    version = versions[0]["version"]
    
    return {"model_name": model_name, "version": version}

@pytest.fixture(scope="module")
def test_features():
    """Fixture to create test feature data."""
    return [
        [10.0, 100.0],  # Example 1
        [8.0, 80.0]     # Example 2
    ]

def test_prediction_with_latest_version(test_models, test_features):
    """Test making a prediction with the latest model version."""
    model_name = test_models["model_name"]
    
    prediction_data = {
        "model_name": model_name,
        "model_version": "latest",
        "features": test_features
    }
    
    response = requests.post(
        f"{INFERENCE_SERVER_URL}/predict",
        json=prediction_data
    )
    
    assert response.status_code == 200, f"Failed to make prediction with latest version: {response.text}"
    
    result = response.json()
    assert "predictions" in result, "Response missing predictions key"
    assert isinstance(result["predictions"], list), "Predictions should be a list"
    assert "model" in result, "Response missing model information"
    assert result["model"]["name"] == model_name, f"Expected model name {model_name}, got {result['model'].get('name')}"

def test_prediction_with_specific_version(test_models, test_features):
    """Test making a prediction with a specific model version."""
    model_name = test_models["model_name"]
    version = test_models["version"]
    
    prediction_data = {
        "model_name": model_name,
        "model_version": version,
        "features": test_features
    }
    
    response = requests.post(
        f"{INFERENCE_SERVER_URL}/predict",
        json=prediction_data
    )
    
    assert response.status_code == 200, f"Failed to make prediction with version {version}: {response.text}"
    
    result = response.json()
    assert "predictions" in result, "Response missing predictions key"
    assert isinstance(result["predictions"], list), "Predictions should be a list"
    assert "model" in result, "Response missing model information"
    assert result["model"]["name"] == model_name, f"Expected model name {model_name}, got {result['model'].get('name')}"
    assert result["model"]["version"] == version, f"Expected version {version}, got {result['model'].get('version')}"

def test_metadata_in_response(test_models, test_features):
    """Test that the prediction response includes model metadata."""
    model_name = test_models["model_name"]
    version = test_models["version"]
    
    prediction_data = {
        "model_name": model_name,
        "model_version": version,
        "features": test_features
    }
    
    response = requests.post(
        f"{INFERENCE_SERVER_URL}/predict",
        json=prediction_data
    )
    
    assert response.status_code == 200, f"Failed to make prediction: {response.text}"
    
    result = response.json()
    assert "model" in result, "Response missing model information"
    assert "metadata" in result["model"], "Response missing model metadata"
    
    metadata = result["model"]["metadata"]
    # Check for expected metadata fields
    assert "training_date" in metadata, "Metadata missing training_date"
    assert "description" in metadata, "Metadata missing description"

def test_batch_predict(test_models):
    """Test the batch prediction endpoint."""
    model_name = test_models["model_name"]
    
    # Create multiple batches of features
    batches = [
        [[10.0, 100.0]],  # Batch 1 with 1 example
        [[8.0, 80.0], [6.0, 60.0]]  # Batch 2 with 2 examples
    ]
    
    prediction_data = {
        "model_name": model_name,
        "model_version": "latest",
        "batches": batches
    }
    
    response = requests.post(
        f"{INFERENCE_SERVER_URL}/batch_predict",
        json=prediction_data
    )
    
    assert response.status_code == 200, f"Failed to make batch prediction: {response.text}"
    
    result = response.json()
    assert "batch_results" in result, "Response missing batch_results key"
    assert len(result["batch_results"]) == len(batches), f"Expected {len(batches)} batch results, got {len(result.get('batch_results', []))}"
    
    # Check each batch result
    for i, batch_result in enumerate(result["batch_results"]):
        assert "batch_index" in batch_result, f"Batch result {i} missing batch_index"
        assert batch_result["batch_index"] == i, f"Expected batch_index {i}, got {batch_result.get('batch_index')}"
        assert "predictions" in batch_result, f"Batch result {i} missing predictions"
        assert "success" in batch_result, f"Batch result {i} missing success indicator"
        assert batch_result["success"] is True, f"Batch {i} prediction failed"

def test_error_handling_invalid_features(test_models):
    """Test error handling for invalid feature data."""
    model_name = test_models["model_name"]
    
    # Invalid features (string instead of numeric)
    prediction_data = {
        "model_name": model_name,
        "model_version": "latest",
        "features": [["invalid", "data"]]
    }
    
    response = requests.post(
        f"{INFERENCE_SERVER_URL}/predict",
        json=prediction_data
    )
    
    # This should return a 500 error since the model can't process string features
    assert response.status_code == 500, "Expected error response for invalid features"
    assert "error" in response.json(), "Error response missing error message"
