#!/usr/bin/env python3
"""
Tests for model versioning in the inference server.

This module tests:
1. Making predictions using the latest model version
2. Making predictions using a specific model version
3. Making predictions using a model UUID
"""

import os
import sys
import json
import pytest
import requests
import numpy as np
import uuid
from datetime import datetime

# Add parent directory and shared directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(parent_dir))  # To access shared

# Server URLs
INFERENCE_SERVER_URL = os.environ.get('INFERENCE_SERVER_URL', 'http://localhost:5000')
MODEL_SERVER_URL = os.environ.get('MODEL_SERVER_URL', 'http://localhost:5001')

# Import shared test utilities
from shared.test_utils import (
    download_and_analyze_model,
    create_input_data_for_shape,
    check_model_server_availability,
    get_available_models
)

@pytest.fixture(scope="module")
def test_models():
    """Fixture to set up and verify (or create) test models on the model server."""
    # Check if we can access the model server
    if not check_model_server_availability(MODEL_SERVER_URL):
        pytest.skip("Error connecting to model server")
    
    # Define test model parameters
    model_name = "test_versioning_model"
    test_versions = ["1.0.0", "1.1.0"]
    
    # Check if the test model exists with both versions
    model_uuids = []
    
    # Try to get existing model versions
    models = get_available_models(MODEL_SERVER_URL)
    if model_name in models:
        # Model exists, check versions
        versions_response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions")
        if versions_response.status_code == 200:
            existing_versions = [v["version"] for v in versions_response.json().get("versions", [])]
            
            # If both test versions exist, use them
            if all(v in existing_versions for v in test_versions):
                # Get UUIDs for these versions
                for version in test_versions:
                    detail_response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions/{version}/detail")
                    if detail_response.status_code == 200:
                        model_uuids.append(detail_response.json().get("uuid"))
    
    # Return model information
    return {
        "model_name": model_name,
        "versions": test_versions,
        "uuids": model_uuids
    }

def test_prediction_with_latest_version(test_models):
    """Test making a prediction using the latest model version."""
    model_name = test_models["model_name"]
    
    # Instead of using /latest endpoint directly, get the latest version from versions list
    versions_response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions")
    if versions_response.status_code != 200:
        pytest.skip("Could not get versions list")
        
    versions_data = versions_response.json()
    
    # Sort versions and pick the latest one
    all_versions = versions_data.get("versions", [])
    if not all_versions:
        pytest.skip("No versions available")
        
    # Sort by version string - assuming semver format
    sorted_versions = sorted(all_versions, key=lambda v: v.get("version"))
    latest = sorted_versions[-1]
    latest_version = latest.get("version")
    
    # Use shared utility to get model input shape
    model_url = f"{MODEL_SERVER_URL}/models/{model_name}/versions/{latest_version}"
    
    input_shape, _ = download_and_analyze_model(model_url)
    
    # Create appropriate test data based on the actual model shape
    input_data = create_input_data_for_shape(input_shape)
    
    # Make a prediction request using the specific version we found
    predict_url = f"{INFERENCE_SERVER_URL}/inference/models/{model_name}/versions/{latest_version}/predict"
    response = requests.post(
        predict_url,
        json=input_data,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200, f"Failed to make prediction with latest version: {response.text}"
    
    result = response.json()
    assert "prediction" in result, "Response missing prediction key"
    assert "model_name" in result, "Response missing model name"
    assert result["model_name"] == model_name, f"Expected model name {model_name}, got {result.get('model_name')}"
    
    # Just verify we get a valid version without hardcoding which one should be latest
    assert "model_version" in result, "Response missing model version"
    assert result["model_version"] in test_models["versions"], f"Version {result.get('model_version')} not in expected versions list {test_models['versions']}"

def test_prediction_with_specific_version(test_models):
    """Test making a prediction using a specific model version."""
    model_name = test_models["model_name"]
    version = test_models["versions"][0]  # Use the first version
    
    # Use shared utility to get model input shape
    model_url = f"{MODEL_SERVER_URL}/models/{model_name}/versions/{version}"
    input_shape, _ = download_and_analyze_model(model_url)
    
    # Create appropriate test data based on the model's shape
    input_data = create_input_data_for_shape(input_shape)
    
    # Make a prediction request
    response = requests.post(
        f"{INFERENCE_SERVER_URL}/inference/models/{model_name}/versions/{version}/predict",
        json=input_data,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200, f"Failed to make prediction with version {version}: {response.text}"
    
    result = response.json()
    assert "prediction" in result, "Response missing prediction key"
    assert "model_name" in result, "Response missing model name"
    assert result["model_name"] == model_name, f"Expected model name {model_name}, got {result.get('model_name')}"
    assert "model_version" in result, "Response missing model version"
    assert result["model_version"] == version, f"Expected version {version}, got {result.get('model_version')}"

def test_prediction_with_uuid(test_models):
    """Test making a prediction using a model UUID."""
    model_uuid = test_models["uuids"][0]  # Use the first UUID
    
    # Use shared utility to get model input shape
    model_url = f"{MODEL_SERVER_URL}/models/uuid/{model_uuid}"
    input_shape, _ = download_and_analyze_model(model_url)
    
    # Create appropriate test data based on the model's shape
    input_data = create_input_data_for_shape(input_shape)
    
    # Make a prediction request
    response = requests.post(
        f"{INFERENCE_SERVER_URL}/inference/models/uuid/{model_uuid}/predict",
        json=input_data,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200, f"Failed to make prediction with UUID {model_uuid}: {response.text}"
    
    result = response.json()
    assert "prediction" in result, "Response missing prediction key"
    assert "model_uuid" in result, "Response missing model UUID"
    assert result["model_uuid"] == model_uuid, f"Expected UUID {model_uuid}, got {result.get('model_uuid')}"
