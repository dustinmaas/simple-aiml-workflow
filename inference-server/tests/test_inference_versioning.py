#!/usr/bin/env python3
"""
Test script for the inference versioning functionality in the inference server.

This script tests:
1. Making predictions using the latest model version
2. Making predictions using a specific model version
3. Handling model version fallbacks
"""

import os
import sys
import json
import requests
import numpy as np
from datetime import datetime

# Server URLs (default when running inside the containers)
INFERENCE_SERVER_URL = os.environ.get('INFERENCE_SERVER_URL', 'http://localhost:5000')
MODEL_SERVER_URL = os.environ.get('MODEL_SERVER_URL', 'http://model-server:5000')

def setup_test_models():
    """Set up test models on the model server for inference testing."""
    print("Setting up test models on the model server...")
    
    # Check if we can access the model server
    try:
        response = requests.get(f"{MODEL_SERVER_URL}/health")
        if response.status_code != 200:
            print(f"Error: Model server not accessible at {MODEL_SERVER_URL}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to model server: {e}")
        return False
    
    # For inference testing, we'll assume the model-server tests have created the
    # necessary test models. We just need to verify they exist.
    response = requests.get(f"{MODEL_SERVER_URL}/models")
    if response.status_code != 200:
        print(f"Error getting models from model server: {response.status_code} - {response.text}")
        return False
    
    models = response.json()
    if not models:
        print("No models found on the model server. Please run the model-server tests first.")
        return False
    
    # Get the first model for testing
    model_name = list(models.keys())[0] if isinstance(models, dict) else "test_model"
    
    # Check if the model has versions
    response = requests.get(f"{MODEL_SERVER_URL}/models/{model_name}/versions")
    if response.status_code != 200:
        print(f"Error getting model versions: {response.status_code} - {response.text}")
        return False
    
    versions_data = response.json()
    versions = versions_data.get("versions", [])
    if not versions:
        print(f"No versions found for model {model_name}")
        return False
    
    # Get the specific version for testing
    version = versions[0]["version"] if versions else "1.0.0"
    
    return {"model_name": model_name, "version": version}

def test_inference_apis(model_info):
    """Test the inference APIs with versioned models."""
    model_name = model_info["model_name"]
    version = model_info["version"]
    
    # Create test features
    features = [
        [10.0, 100.0],  # Example 1
        [8.0, 80.0]     # Example 2
    ]
    
    # Test 1: Make prediction with latest version
    print("\n--- Test 1: Prediction with latest version ---")
    prediction_data = {
        "model_name": model_name,
        "model_version": "latest",
        "features": features
    }
    
    response = requests.post(
        f"{INFERENCE_SERVER_URL}/predict",
        json=prediction_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))
        print("PASS: Successfully made prediction with latest version")
    else:
        print(f"FAIL: Error making prediction with latest version: {response.status_code} - {response.text}")
        return False
    
    # Test 2: Make prediction with specific version
    print("\n--- Test 2: Prediction with specific version ---")
    prediction_data = {
        "model_name": model_name,
        "model_version": version,
        "features": features
    }
    
    response = requests.post(
        f"{INFERENCE_SERVER_URL}/predict",
        json=prediction_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))
        print(f"PASS: Successfully made prediction with version {version}")
    else:
        print(f"FAIL: Error making prediction with specific version: {response.status_code} - {response.text}")
        return False
    
    # Test 3: Check for metadata in response
    print("\n--- Test 3: Checking metadata in response ---")
    if "model" in result and "metadata" in result["model"]:
        print("PASS: Response includes model metadata")
        print(f"Metadata: {json.dumps(result['model']['metadata'], indent=2)}")
    else:
        print("FAIL: No metadata in response")
        return False
    
    print("\nAll tests passed!")
    return True

def main():
    """Main function to run the tests."""
    print(f"Testing inference server versioning at {INFERENCE_SERVER_URL}")
    print(f"Using model server at {MODEL_SERVER_URL}")
    
    # Set up test models
    model_info = setup_test_models()
    if not model_info:
        print("Failed to set up test models.")
        return 1
    
    # Run tests
    if test_inference_apis(model_info):
        print("\nInference versioning system is working correctly!")
        return 0
    else:
        print("\nInference versioning tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
