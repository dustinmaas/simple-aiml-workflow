#!/usr/bin/env python3
"""
Model Server API for ONNX models.

This API serves versioned ONNX models to other services, particularly the inference server.
It implements a simple file-based versioning system for models.
"""

import os
import sys
import logging
import json
import re
import datetime
from typing import Dict, List, Optional, Tuple, Any
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import onnxruntime as ort
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = os.environ.get('MODELS_DIR', '/app/models')

# Version regex pattern for model filenames: name_v1.0.0.onnx
VERSION_PATTERN = re.compile(r'^(.+)_v(\d+)\.(\d+)\.(\d+)\.onnx$')

# Helper functions for metadata handling
def get_metadata_path(model_path: str) -> str:
    """Get the path to the metadata file for a given model path."""
    return model_path.replace('.onnx', '.metadata.json')

def save_metadata(model_path: str, metadata: Dict[str, Any]) -> None:
    """Save metadata to a JSON file alongside the model."""
    metadata_path = get_metadata_path(model_path)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_metadata(model_path: str) -> Dict[str, Any]:
    """Load metadata from a JSON file if it exists."""
    metadata_path = get_metadata_path(model_path)
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}

app = Flask(__name__)

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def parse_model_filename(filename: str) -> Optional[Tuple[str, Tuple[int, int, int]]]:
    """
    Parse a model filename to extract model name and version.
    
    Args:
        filename: Model filename in format "name_v1.0.0.onnx"
        
    Returns:
        Tuple of (model_name, (major, minor, patch)) or None if not matching pattern
    """
    match = VERSION_PATTERN.match(filename)
    if not match:
        return None
    
    model_name = match.group(1)
    version = (int(match.group(2)), int(match.group(3)), int(match.group(4)))
    return model_name, version

def get_model_versions(model_name: str) -> List[Tuple[str, Tuple[int, int, int]]]:
    """
    Get all versions of a specific model.
    
    Args:
        model_name: Base name of the model
        
    Returns:
        List of tuples (filename, (major, minor, patch)) sorted by version
    """
    versions = []
    
    for filename in os.listdir(MODELS_DIR):
        parsed = parse_model_filename(filename)
        if parsed and parsed[0] == model_name:
            versions.append((filename, parsed[1]))
    
    # Sort by version (major, minor, patch)
    return sorted(versions, key=lambda x: x[1])

def get_latest_version(model_name: str) -> Optional[str]:
    """
    Get the latest version of a model.
    
    Args:
        model_name: Base name of the model
        
    Returns:
        Filename of the latest version or None if no versions found
    """
    versions = get_model_versions(model_name)
    if not versions:
        return None
    
    # Return the filename of the latest version
    return versions[-1][0]

def version_to_string(version: Tuple[int, int, int]) -> str:
    """Convert version tuple to string."""
    return f"{version[0]}.{version[1]}.{version[2]}"

def group_models_by_name() -> Dict[str, List[Dict]]:
    """
    Group all models by base name with their versions.
    
    Returns:
        Dictionary mapping model names to lists of version information
    """
    result = {}
    
    for filename in os.listdir(MODELS_DIR):
        parsed = parse_model_filename(filename)
        if parsed:
            model_name, version = parsed
            
            if model_name not in result:
                result[model_name] = []
                
            version_str = version_to_string(version)
            result[model_name].append({
                "version": version_str,
                "filename": filename,
                "path": os.path.join(MODELS_DIR, filename)
            })
    
    # Sort versions within each model
    for model_name in result:
        result[model_name].sort(key=lambda x: tuple(map(int, x["version"].split("."))))
    
    return result

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models grouped by name with versions."""
    try:
        grouped_models = group_models_by_name()
        return jsonify(grouped_models)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models/<model_name>/versions', methods=['GET'])
def list_model_versions(model_name):
    """List all versions of a specific model."""
    try:
        model_versions = get_model_versions(model_name)
        versions = []
        
        for filename, version_tuple in model_versions:
            version_str = version_to_string(version_tuple)
            versions.append({
                "version": version_str,
                "filename": filename,
                "path": os.path.join(MODELS_DIR, filename)
            })
            
        return jsonify({"model": model_name, "versions": versions})
    except Exception as e:
        logger.error(f"Error listing model versions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models/<model_name>/versions/latest', methods=['GET'])
def get_latest_model_version(model_name):
    """Get the latest version of a model."""
    try:
        latest_version = get_latest_version(model_name)
        if not latest_version:
            return jsonify({"error": f"No versions found for model {model_name}"}), 404
        
        filepath = os.path.join(MODELS_DIR, latest_version)
        
        # Extract version from filename
        parsed = parse_model_filename(latest_version)
        if not parsed:
            return jsonify({"error": "Invalid model filename format"}), 500
        
        version_str = version_to_string(parsed[1])
        
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        logger.error(f"Error getting latest model version: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models/<model_name>/versions/<version>', methods=['GET'])
def get_specific_model_version(model_name, version):
    """Get a specific version of a model."""
    try:
        # Format the expected filename based on model name and version
        filename = f"{model_name}_v{version}.onnx"
        filepath = os.path.join(MODELS_DIR, filename)
        
        if not os.path.exists(filepath):
            return jsonify({"error": f"Model {model_name} version {version} not found"}), 404
        
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        logger.error(f"Error getting model version: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models/<model_name>', methods=['GET'])
def get_model(model_name):
    """Retrieve the latest version of a specific model."""
    return get_latest_model_version(model_name)

@app.route('/models/<model_name>/versions/<version>/metadata', methods=['GET'])
def get_model_metadata(model_name, version):
    """Get metadata for a specific model version."""
    filename = f"{model_name}_v{version}.onnx"
    filepath = os.path.join(MODELS_DIR, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": f"Model {model_name} version {version} not found"}), 404
    
    metadata = load_metadata(filepath)
    return jsonify(metadata)

@app.route('/models/<model_name>/versions/<version>', methods=['POST'])
def upload_model_version(model_name, version):
    """Upload and store a specific version of a model."""
    if 'model' not in request.files:
        return jsonify({"error": "No model file provided"}), 400
    
    model_file = request.files['model']
    if model_file.filename == '':
        return jsonify({"error": "Empty model filename"}), 400
    
    # Create filename with versioning
    filename = f"{model_name}_v{version}.onnx"
    filepath = os.path.join(MODELS_DIR, filename)
    
    # Check if this version already exists
    if os.path.exists(filepath):
        return jsonify({"error": f"Version {version} of model {model_name} already exists"}), 409
    
    # Save the model file
    model_file.save(filepath)
    
    # Extract metadata from form data if provided
    metadata = {}
    if 'metadata' in request.form:
        try:
            metadata = json.loads(request.form['metadata'])
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in metadata form field, ignoring metadata")
    
    # Add version info to metadata
    metadata['version'] = version
    metadata['model_name'] = model_name
    metadata['upload_time'] = datetime.datetime.now().isoformat()
    
    # Save metadata to a separate file
    save_metadata(filepath, metadata)
    
    # Validate model
    try:
        session = ort.InferenceSession(filepath)
        test_input_data = np.array([[10.0, 90.0]], dtype=np.float32)  # Example input (batch size 1, 2 features)

        # Run inference
        outputs = session.run(None, {"input": test_input_data})  # 'input' is the input name used during export
        del session
        logger.info(f"Model {model_name} version {version} uploaded successfully")
        logger.info(f"Test inputs/outputs: {test_input_data} -> {outputs}")
        return jsonify({
            "success": True, 
            "message": f"Model {model_name} version {version} uploaded successfully",
            "model": {
                "name": model_name,
                "version": version,
                "path": filepath,
                "metadata": metadata
            }
        }), 200
    except Exception as e:
        # If validation fails, remove the model and metadata files
        if os.path.exists(filepath):
            os.remove(filepath)
        metadata_path = get_metadata_path(filepath)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        return jsonify({"error": f"Invalid model file: {str(e)}"}), 400

@app.route('/models/<model_name>/versions/<version>', methods=['DELETE'])
def delete_model_version(model_name, version):
    """Delete a specific version of a model."""
    filename = f"{model_name}_v{version}.onnx"
    filepath = os.path.join(MODELS_DIR, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": f"Model {model_name} version {version} not found"}), 404
    
    try:
        # Delete the model file
        os.remove(filepath)
        
        # Also delete the metadata file if it exists
        metadata_path = get_metadata_path(filepath)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            logger.info(f"Deleted metadata file: {metadata_path}")
            
        return jsonify({
            "message": f"Model {model_name} version {version} deleted successfully"
        })
    except Exception as e:
        logger.error(f"Error deleting model version: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models/<model_name>', methods=['POST'])
def upload_model(model_name):
    """Upload model and automatically assign the next patch version."""
    if 'model' not in request.files:
        return jsonify({"error": "No model file provided"}), 400
    
    model_file = request.files['model']
    if model_file.filename == '':
        return jsonify({"error": "Empty model filename"}), 400
    
    # Get all existing versions of this model
    versions = get_model_versions(model_name)
    
    if not versions:
        # No versions exist, start with 1.0.0
        new_version = (1, 0, 0)
    else:
        # Get the latest version and increment the patch version
        latest_filename, latest_version = versions[-1]
        new_version = (latest_version[0], latest_version[1], latest_version[2] + 1)
    
    # Create version string for the response
    version_str = version_to_string(new_version)
    
    # Redirect to version-specific upload endpoint
    return upload_model_version(model_name, version_str)

@app.route('/models/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    """Delete all versions of a model."""
    versions = get_model_versions(model_name)
    
    if not versions:
        return jsonify({"error": f"No versions found for model {model_name}"}), 404
    
    deleted_count = 0
    errors = []
    
    for filename, _ in versions:
        filepath = os.path.join(MODELS_DIR, filename)
        try:
            # Delete the model file
            os.remove(filepath)
            
            # Also delete the metadata file if it exists
            metadata_path = get_metadata_path(filepath)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                logger.info(f"Deleted metadata file: {metadata_path}")
                
            deleted_count += 1
        except Exception as e:
            errors.append(f"Failed to delete {filename}: {str(e)}")
    
    if errors:
        return jsonify({
            "message": f"Partially deleted model {model_name}. Deleted {deleted_count} of {len(versions)} versions.",
            "errors": errors
        }), 207  # Multi-Status
    
    return jsonify({
        "message": f"Successfully deleted all {deleted_count} versions of model {model_name}"
    })

def load_model(model_name, version=None):
    """
    Load model from disk and return it with metadata.
    
    Args:
        model_name: Base name of the model
        version: Specific version to load. If None, loads the latest version.
    
    Returns:
        Tuple of (ONNX Runtime InferenceSession, metadata dict)
    """
    if version:
        # Load specific version
        filepath = os.path.join(MODELS_DIR, f"{model_name}_v{version}.onnx")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
    else:
        # Load latest version
        latest_version_filename = get_latest_version(model_name)
        if not latest_version_filename:
            raise FileNotFoundError(f"No versions found for model {model_name}")
        
        filepath = os.path.join(MODELS_DIR, latest_version_filename)
    
    try:
        # Load the ONNX model
        model = ort.InferenceSession(filepath)
        
        # Load associated metadata
        metadata = load_metadata(filepath)
        
        # Return both model and metadata
        return model, metadata
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
