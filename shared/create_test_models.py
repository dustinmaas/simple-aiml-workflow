#!/usr/bin/env python3
"""
Unified helper script to create test models for all server tests.

This script consolidates duplicate code from inference-server and model-server
to create test models using the shared ML utilities and upload them via REST API.
"""

import os
import sys
import requests
import torch
import argparse
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path if needed
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import ML utilities with relative or absolute import
try:
    # Try relative import first (when imported as module)
    from .ml_utils import create_and_train_model, export_model_to_onnx, get_default_metadata
except ImportError:
    # Fall back to absolute import (when run as script)
    from ml_utils import create_and_train_model, export_model_to_onnx, get_default_metadata

def create_test_model(model_server_url: str) -> Optional[Dict[str, str]]:
    """Create a single test model and upload it to the model server.
    
    Args:
        model_server_url: URL of the model server
        
    Returns:
        Dict with model info or None if upload failed
    """
    # Model parameters
    model_name = "test_inference_model"
    version = "1.0.0"
    
    # Create and train the model
    model = create_and_train_model()
    
    # Create temp file for ONNX model
    temp_model_path = "/tmp/test_inference_model.onnx"
    
    # Export the model to ONNX format
    export_model_to_onnx(model, temp_model_path)
    
    # Upload model to model server
    with open(temp_model_path, 'rb') as f:
        # Get metadata with model name and version
        metadata = get_default_metadata(model_name, version)
        
        upload_response = requests.post(
            f"{model_server_url}/models/{model_name}/versions/{version}",
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

def create_versioned_test_models(model_server_url: str) -> List[Dict[str, str]]:
    """Create multiple versions of a model for versioning tests.
    
    Args:
        model_server_url: URL of the model server
        
    Returns:
        List of dicts with model info
    """
    model_name = "test_versioning_model"
    versions = ["1.0.0", "1.1.0"]
    results = []
    
    # Create and train base model
    model = create_and_train_model()
    
    for version in versions:
        # Export model to temp file with version in filename
        temp_model_path = f"/tmp/test_model_{version}.onnx"
        
        # Make slight parameter adjustments for different versions
        if version != "1.0.0":
            # Slightly modify weights for different versions
            with torch.no_grad():
                model.linear.weight.data = model.linear.weight.data * 1.05
                model.linear.bias.data = model.linear.bias.data + 0.1
        
        # Export to ONNX using the shared utility
        export_model_to_onnx(model, temp_model_path)
        
        # Upload to model server
        with open(temp_model_path, 'rb') as f:
            description = f'Linear regression model version {version} for PRB prediction'
            metadata = get_default_metadata(model_name, version, description)
            
            upload_response = requests.post(
                f"{model_server_url}/models/{model_name}/versions/{version}",
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

def main():
    """Main function to parse arguments and create models."""
    parser = argparse.ArgumentParser(description='Create test models for server tests')
    parser.add_argument('--url', type=str, help='Model server URL')
    parser.add_argument('--simple', action='store_true', help='Create only a simple test model')
    parser.add_argument('--versioned', action='store_true', help='Create only versioned test models')
    args = parser.parse_args()
    
    # If no URL is provided, attempt to use environment variables from constants
    if not args.url:
        try:
            from model_server.utils.constants import MODEL_SERVER_URL
            model_server_url = MODEL_SERVER_URL
        except ImportError:
            try:
                from inference_server.utils.constants import MODEL_SERVER_URL
                model_server_url = MODEL_SERVER_URL
            except ImportError:
                model_server_url = os.environ.get('MODEL_SERVER_URL', 'http://model-server:80')
    else:
        model_server_url = args.url
    
    # Default behavior: create both if neither flag is specified
    create_simple = args.simple or not (args.simple or args.versioned)
    create_versioned = args.versioned or not (args.simple or args.versioned)
    
    # Check if model server is available
    try:
        response = requests.get(f"{model_server_url}/health", timeout=5)
        if response.status_code == 200:
            results = []
            
            # Create a simple test model
            if create_simple:
                simple_model = create_test_model(model_server_url)
                if simple_model:
                    print(f"Test model created: {simple_model}")
                    results.append(simple_model)
                else:
                    print("Failed to create simple test model")
            
            # Create versioned models for versioning tests
            if create_versioned:
                versioned_models = create_versioned_test_models(model_server_url)
                if versioned_models:
                    print(f"Created {len(versioned_models)} versioned test models")
                    results.extend(versioned_models)
                else:
                    print("Failed to create versioned test models")
            
            return results
        else:
            print(f"Model server health check failed: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to model server: {e}")
        return []

if __name__ == "__main__":
    main()
