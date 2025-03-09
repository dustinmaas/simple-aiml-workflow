#!/usr/bin/env python3
"""
Model management routes for the model server.

This module defines routes for model management operations, including:
- Listing available models and versions
- Retrieving specific model versions
- Uploading new models and versions
- Deleting models and versions
"""

import os
import logging
import json
from typing import Dict, Any
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename

from utils.constants import MODELS_DIR
from utils.version_manager import (
    get_model_versions,
    get_latest_version,
    group_models_by_name,
    version_to_string,
    get_next_version
)
from utils.metadata_manager import (
    save_metadata,
    load_metadata,
    clean_metadata,
    merge_metadata,
    enriched_metadata_response
)
from utils.model_validator import (
    validate_onnx_model,
    get_model_summary
)

logger = logging.getLogger(__name__)

# Create Blueprint for model routes
model_bp = Blueprint('models', __name__, url_prefix='/models')

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

@model_bp.route('', methods=['GET'])
def list_models():
    """
    List all available models grouped by name with versions.
    
    Returns:
        JSON response with all models and their versions
    """
    try:
        grouped_models = group_models_by_name(MODELS_DIR)
        return jsonify(grouped_models)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({"error": str(e)}), 500

@model_bp.route('/<model_name>/versions', methods=['GET'])
def list_model_versions(model_name):
    """
    List all versions of a specific model.
    
    Args:
        model_name: Base name of the model
        
    Returns:
        JSON response with all versions of the model
    """
    try:
        model_versions = get_model_versions(model_name, MODELS_DIR)
        versions = []
        
        for filename, version_tuple in model_versions:
            version_str = version_to_string(version_tuple)
            filepath = os.path.join(MODELS_DIR, filename)
            versions.append({
                "version": version_str,
                "filename": filename,
                "path": filepath,
                "size_bytes": os.path.getsize(filepath),
                "created": os.path.getctime(filepath)
            })
            
        return jsonify({"model": model_name, "versions": versions})
    except Exception as e:
        logger.error(f"Error listing model versions: {e}")
        return jsonify({"error": str(e)}), 500

@model_bp.route('/<model_name>/versions/latest', methods=['GET'])
def get_latest_model_version(model_name):
    """
    Get the latest version of a model.
    
    Args:
        model_name: Base name of the model
        
    Returns:
        The model file as attachment
    """
    try:
        latest_version = get_latest_version(model_name, MODELS_DIR)
        if not latest_version:
            return jsonify({"error": f"No versions found for model {model_name}"}), 404
        
        filepath = os.path.join(MODELS_DIR, latest_version)
        
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        logger.error(f"Error getting latest model version: {e}")
        return jsonify({"error": str(e)}), 500

@model_bp.route('/<model_name>/versions/<version>', methods=['GET'])
def get_specific_model_version(model_name, version):
    """
    Get a specific version of a model.
    
    Args:
        model_name: Base name of the model
        version: Specific version to retrieve
        
    Returns:
        The model file as attachment
    """
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

@model_bp.route('/<model_name>', methods=['GET'])
def get_model(model_name):
    """
    Retrieve the latest version of a specific model.
    
    Args:
        model_name: Base name of the model
        
    Returns:
        The latest model version as attachment
    """
    return get_latest_model_version(model_name)

@model_bp.route('/<model_name>/versions/<version>/metadata', methods=['GET'])
def get_model_metadata(model_name, version):
    """
    Get metadata for a specific model version.
    
    Args:
        model_name: Base name of the model
        version: Specific version to retrieve metadata for
        
    Returns:
        JSON response with model metadata
    """
    filename = f"{model_name}_v{version}.onnx"
    filepath = os.path.join(MODELS_DIR, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": f"Model {model_name} version {version} not found"}), 404
    
    metadata = load_metadata(filepath)
    metadata = enriched_metadata_response(metadata)
    
    return jsonify(metadata)

@model_bp.route('/<model_name>/versions/<version>', methods=['POST'])
def upload_model_version(model_name, version):
    """
    Upload and store a specific version of a model.
    
    Args:
        model_name: Base name of the model
        version: Specific version string
        
    Returns:
        JSON response with upload result
    """
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
    
    # Validate model
    try:
        # Run model validation
        is_valid, validation_info = validate_onnx_model(filepath)
        
        if not is_valid:
            # If validation fails, remove the model file
            os.remove(filepath)
            return jsonify({"error": f"Invalid model file: {validation_info.get('error', 'Unknown error')}"}), 400
        
        # Add validation info to metadata
        metadata['validation_info'] = validation_info
        
        # Save metadata to a separate file
        save_metadata(filepath, metadata)
        
        logger.info(f"Model {model_name} version {version} uploaded successfully")
        
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
        # If any error occurs, clean up files
        if os.path.exists(filepath):
            os.remove(filepath)
        clean_metadata(filepath)
        
        logger.error(f"Error during model upload: {e}")
        return jsonify({"error": f"Error processing model: {str(e)}"}), 500

@model_bp.route('/<model_name>', methods=['POST'])
def upload_model(model_name):
    """
    Upload model and automatically assign the next patch version.
    
    Args:
        model_name: Base name of the model
        
    Returns:
        JSON response with upload result
    """
    # Get the next version number
    new_version = get_next_version(model_name, MODELS_DIR)
    version_str = version_to_string(new_version)
    
    # Use the version-specific upload endpoint
    return upload_model_version(model_name, version_str)

@model_bp.route('/<model_name>/versions/<version>', methods=['DELETE'])
def delete_model_version(model_name, version):
    """
    Delete a specific version of a model.
    
    Args:
        model_name: Base name of the model
        version: Specific version to delete
        
    Returns:
        JSON response with deletion result
    """
    filename = f"{model_name}_v{version}.onnx"
    filepath = os.path.join(MODELS_DIR, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": f"Model {model_name} version {version} not found"}), 404
    
    try:
        # Delete the model file
        os.remove(filepath)
        
        # Also delete the metadata file if it exists
        clean_metadata(filepath)
            
        return jsonify({
            "message": f"Model {model_name} version {version} deleted successfully"
        })
    except Exception as e:
        logger.error(f"Error deleting model version: {e}")
        return jsonify({"error": str(e)}), 500

@model_bp.route('/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    """
    Delete all versions of a model.
    
    Args:
        model_name: Base name of the model
        
    Returns:
        JSON response with deletion result
    """
    versions = get_model_versions(model_name, MODELS_DIR)
    
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
            clean_metadata(filepath)
            
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

@model_bp.route('/<model_name>/versions/<version>/detail', methods=['GET'])
def get_model_detail(model_name, version):
    """
    Get detailed information about a specific model version.
    
    Args:
        model_name: Base name of the model
        version: Specific version to get details for
        
    Returns:
        JSON response with detailed model information
    """
    filename = f"{model_name}_v{version}.onnx"
    filepath = os.path.join(MODELS_DIR, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": f"Model {model_name} version {version} not found"}), 404
    
    try:
        # Get model summary
        summary = get_model_summary(filepath)
        
        # Load metadata
        metadata = load_metadata(filepath)
        
        # Combine information
        response = {
            "model_name": model_name,
            "version": version,
            "filename": filename,
            "path": filepath,
            "size_bytes": summary["size_bytes"],
            "is_valid": summary["is_valid"],
            "validation_info": summary["validation_info"],
            "metadata": metadata,
            "filesystem_info": {
                "created": os.path.getctime(filepath),
                "modified": os.path.getmtime(filepath),
                "permissions": oct(os.stat(filepath).st_mode)[-3:]
            }
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error getting model details: {e}")
        return jsonify({"error": str(e)}), 500
