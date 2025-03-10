#!/usr/bin/env python3
"""
Model management routes using UUID-based storage.

This module defines routes for model management operations including:
- Listing available models and versions
- Retrieving specific model versions
- Uploading new models and versions
- Deleting models and versions
"""

import os
import logging
import json
import uuid
from typing import Dict, Any, List, Optional
from flask import Blueprint, request, jsonify, send_file, current_app
import io

from utils.constants import MODEL_DB_PATH, MODEL_STORAGE_DIR
from utils.database import ModelDatabase
from utils.storage import ModelStorage
from utils.model_validator import validate_onnx_model, get_model_summary

logger = logging.getLogger(__name__)

# Create Blueprint for model routes
model_bp = Blueprint('models', __name__, url_prefix='/models')

# Initialize database and storage
db = ModelDatabase(MODEL_DB_PATH)
storage = ModelStorage(MODEL_STORAGE_DIR)

@model_bp.route('', methods=['GET'])
def list_models():
    """
    List all available models grouped by name with versions.

    Returns:
        JSON response with all models and their versions
    """
    try:
        grouped_models = db.list_models()
        
        # Format the response to match the existing API structure
        result = {}
        for name, versions in grouped_models.items():
            result[name] = []
            for model in versions:
                result[name].append({
                    'uuid': model['uuid'],
                    'version': model['version'],
                    'path': model['file_path'],
                    'filename': f"{name}_v{model['version']}.onnx"
                })
        
        return jsonify(result)
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
        grouped_models = db.list_models()
        
        if model_name not in grouped_models:
            return jsonify({"error": f"Model {model_name} not found"}), 404
        
        # Format the response to include detailed version info
        versions = []
        for model in grouped_models[model_name]:
            versions.append({
                'uuid': model['uuid'],
                'version': model['version'],
                'path': model['file_path'],
                'filename': f"{model_name}_v{model['version']}.onnx",
                'size_bytes': model['file_size'],
                'created': model['created_at']
            })
        
        # Sort versions by creation time, newest first
        versions.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            "model": model_name,
            "versions": versions
        })
    except Exception as e:
        logger.error(f"Error listing model versions: {e}")
        return jsonify({"error": str(e)}), 500

@model_bp.route('/<model_name>/versions/<version>', methods=['GET'])
def get_model_by_name_version(model_name, version):
    """
    Get a specific version of a model.

    Args:
        model_name: Base name of the model
        version: Specific version to retrieve
        
    Returns:
        The model file as attachment
    """
    try:
        try:
            model_info = db.get_model_by_name_version(model_name, version)
        except ModelNotFoundError:
            return jsonify({"error": f"Model {model_name} version {version} not found"}), 404

        # Extract the storage UUID from the file path
        # Format of file_path is: "/data/models/[storage_uuid].onnx"
        file_path = model_info["file_path"]
        try:
            storage_uuid = os.path.basename(file_path).replace(".onnx", "")
            logger.info(f"Database UUID: {model_info['uuid']}, Storage UUID: {storage_uuid}")
        except Exception as e:
            logger.error(f"Failed to extract storage UUID from path {file_path}: {e}")
            storage_uuid = model_info["uuid"]  # Fallback to database UUID

        model_data = storage.get_model(storage_uuid)
        if not model_data:
            return jsonify({"error": f"Model file not found for {model_name} version {version}"}), 404

        filename = f"{model_name}_v{version}.onnx"

        return send_file(
            io.BytesIO(model_data),
            mimetype='application/octet-stream',
            as_attachment=True,
            attachment_filename=filename
        )
    except ModelNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error getting model version: {e}")
        return jsonify({"error": "Internal server error", "message": "An unexpected error occurred."}), 500

@model_bp.route('/<model_name>', methods=['GET'])
def get_model_versions(model_name):
    """
    Get all versions of a model.

    Args:
        model_name: Base name of the model
        
    Returns:
        JSON response with all versions of the model
    """
    return list_model_versions(model_name)

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
        model_info = db.get_latest_model_by_name(model_name)
        if not model_info:
            return jsonify({"error": f"No versions found for model {model_name}"}), 404

        # Extract the storage UUID from the file path
        # Format of file_path is: "/data/models/[storage_uuid].onnx"
        file_path = model_info["file_path"]
        try:
            storage_uuid = os.path.basename(file_path).replace(".onnx", "")
            logger.info(f"Database UUID: {model_info['uuid']}, Storage UUID: {storage_uuid}")
        except Exception as e:
            logger.error(f"Failed to extract storage UUID from path {file_path}: {e}")
            storage_uuid = model_info["uuid"]  # Fallback to database UUID

        model_data = storage.get_model(storage_uuid)
        if not model_data:
            return jsonify({"error": f"Model file not found for {model_name}"}), 404

        filename = f"{model_name}_v{model_info['version']}.onnx"

        return send_file(
            io.BytesIO(model_data),
            mimetype='application/octet-stream',
            as_attachment=True,
            attachment_filename=filename
        )
    except Exception as e:
        logger.error(f"Error getting latest model version: {e}")
        return jsonify({"error": str(e)}), 500

@model_bp.route('/<model_name>/versions/<version>/metadata', methods=['GET'])
def get_model_metadata(model_name, version):
    """
    Get metadata for a specific model version.

    Args:
        model_name: Base name of the model
        version: Specific version to get metadata for
        
    Returns:
        JSON response with model metadata
    """
    try:
        model_info = db.get_model_by_name_version(model_name, version)
        if not model_info:
            return jsonify({"error": f"Model {model_name} version {version} not found"}), 404

        metadata = db.get_metadata(model_info['uuid'])
        if not metadata:
            return jsonify({})  # Return empty metadata if none exists
            
        # Add upload time in a readable format
        if 'upload_time' in metadata:
            try:
                from datetime import datetime
                timestamp = datetime.fromisoformat(metadata['upload_time'])
                metadata['upload_time_formatted'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                # Keep original if we can't parse it
                pass
        
        return jsonify(metadata)
    except Exception as e:
        logger.error(f"Error getting model metadata: {e}")
        return jsonify({"error": str(e)}), 500

@model_bp.route('/uuid/<uuid>/metadata', methods=['GET'])
def get_model_metadata_by_uuid(uuid):
    """
    Get metadata for a model by UUID.

    Args:
        uuid: UUID of the model
        
    Returns:
        JSON response with model metadata
    """
    try:
        metadata = db.get_metadata(uuid)
        if not metadata:
            return jsonify({"error": f"Metadata not found for model with UUID {uuid}"}), 404
            
        # Add upload time in a readable format
        if 'upload_time' in metadata:
            try:
                from datetime import datetime
                timestamp = datetime.fromisoformat(metadata['upload_time'])
                metadata['upload_time_formatted'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                # Keep original if we can't parse it
                pass
        
        return jsonify(metadata)
    except Exception as e:
        logger.error(f"Error getting model metadata by UUID: {e}")
        return jsonify({"error": str(e)}), 500

from utils.database import ModelExistsError, ModelNotFoundError

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
    # Check if a model file was uploaded
    if 'model' not in request.files:
        return jsonify({"error": "No model file provided"}), 400
        
    model_file = request.files['model']
    if model_file.filename == '':
        return jsonify({"error": "Empty model filename"}), 400
        
    try:
        # Read the model file
        model_data = model_file.read()
        
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
        
        # Generate a single UUID for both storage and database
        single_uuid = str(uuid.uuid4())
        logger.info(f"Generated single UUID {single_uuid} for model {model_name} version {version}")
        
        # Store the model file with this UUID
        file_path, file_size = storage.store_model(single_uuid, model_data)
        
        try:
            # Add to the database with the same UUID
            db_result = db.add_model_with_uuid(single_uuid, model_name, version, file_path, file_size)
            
            # Add metadata to the database
            db.add_metadata(single_uuid, metadata)
            
            return jsonify({
                "success": True,
                "message": f"Model {model_name} version {version} uploaded successfully",
                "uuid": single_uuid,
                "size": file_size
            })
        except ModelExistsError:
            # If db insert failed due to duplicate, clean up the stored file
            storage.delete_model(single_uuid)
            return jsonify({"error": f"Version {version} of model {model_name} already exists"}), 409
        except Exception as e:
            # If db insert failed for other reasons, clean up the stored file
            storage.delete_model(single_uuid)
            logger.error(f"Database operation failed: {e}")
            raise
    except ModelNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        return jsonify({"error": "Internal server error", "message": "An unexpected error occurred."}), 500

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
    try:
        model_info = db.get_model_by_name_version(model_name, version)
        if not model_info:
            return jsonify({"error": f"Model {model_name} version {version} not found"}), 404
            
        model_uuid = model_info['uuid']
        
        # Extract storage UUID from file path
        file_path = model_info["file_path"]
        try:
            storage_uuid = os.path.basename(file_path).replace(".onnx", "")
            logger.info(f"Database UUID: {model_uuid}, Storage UUID: {storage_uuid}")
        except Exception as e:
            logger.error(f"Failed to extract storage UUID from path {file_path}: {e}")
            storage_uuid = model_uuid  # Fallback to database UUID
        
        # Delete from storage
        storage.delete_model(storage_uuid)
        
        # Delete from database
        db.delete_model(model_uuid)
        
        return jsonify({
            "message": f"Model {model_name} version {version} deleted successfully",
            "uuid": model_uuid
        })
    except Exception as e:
        logger.error(f"Error deleting model version: {e}")
        return jsonify({"error": str(e)}), 500

@model_bp.route('/uuid/<uuid>', methods=['DELETE'])
def delete_model_by_uuid(uuid):
    """
    Delete a model by UUID.

    Args:
        uuid: UUID of the model
        
    Returns:
        JSON response with deletion result
    """
    try:
        model_info = db.get_model_by_uuid(uuid)
        if not model_info:
            return jsonify({"error": f"Model with UUID {uuid} not found"}), 404
            
        # Extract storage UUID from file path
        file_path = model_info["file_path"]
        try:
            storage_uuid = os.path.basename(file_path).replace(".onnx", "")
            logger.info(f"Database UUID: {uuid}, Storage UUID: {storage_uuid}")
        except Exception as e:
            logger.error(f"Failed to extract storage UUID from path {file_path}: {e}")
            storage_uuid = uuid  # Fallback to database UUID
        
        # Delete from storage
        storage.delete_model(storage_uuid)
        
        # Delete from database
        db.delete_model(uuid)
        
        return jsonify({
            "message": f"Model with UUID {uuid} deleted successfully",
            "model_name": model_info['name'],
            "version": model_info['version']
        })
    except Exception as e:
        logger.error(f"Error deleting model by UUID: {e}")
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
    try:
        # Get all versions of the model
        grouped_models = db.list_models()
        
        if model_name not in grouped_models:
            return jsonify({"error": f"Model {model_name} not found"}), 404
            
        deleted_count = 0
        for model in grouped_models[model_name]:
            # Extract storage UUID from file path
            file_path = model["file_path"]
            try:
                storage_uuid = os.path.basename(file_path).replace(".onnx", "")
                logger.info(f"Database UUID: {model['uuid']}, Storage UUID: {storage_uuid}")
            except Exception as e:
                logger.error(f"Failed to extract storage UUID from path {file_path}: {e}")
                storage_uuid = model['uuid']  # Fallback to database UUID
            
            # Delete from storage
            storage.delete_model(storage_uuid)
            
            # Delete from database
            db.delete_model(model['uuid'])
            
            deleted_count += 1
            
        return jsonify({
            "message": f"All versions of model {model_name} deleted successfully",
            "deleted_count": deleted_count
        })
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        return jsonify({"error": str(e)}), 500

@model_bp.route('/uuid/<uuid>', methods=['GET'])
def get_model_by_uuid(uuid):
    """
    Get a model by UUID.

    Args:
        uuid: UUID of the model
        
    Returns:
        The model file as attachment
    """
    try:
        model_info = db.get_model_by_uuid(uuid)
        if not model_info:
            return jsonify({"error": f"Model with UUID {uuid} not found"}), 404

        # Extract the storage UUID from the file path
        # Format of file_path is: "/data/models/[storage_uuid].onnx"
        file_path = model_info["file_path"]
        try:
            storage_uuid = os.path.basename(file_path).replace(".onnx", "")
            logger.info(f"Database UUID: {uuid}, Storage UUID: {storage_uuid}")
        except Exception as e:
            logger.error(f"Failed to extract storage UUID from path {file_path}: {e}")
            storage_uuid = uuid  # Fallback to database UUID

        model_data = storage.get_model(storage_uuid)
        if not model_data:
            return jsonify({"error": f"Model file not found for UUID {uuid}"}), 404

        filename = f"{model_info['name']}_v{model_info['version']}.onnx"

        return send_file(
            io.BytesIO(model_data),
            mimetype='application/octet-stream',
            as_attachment=True,
            attachment_filename=filename
        )
    except Exception as e:
        logger.error(f"Error getting model by UUID: {e}")
        return jsonify({"error": str(e)}), 500

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
    try:
        model_info = db.get_model_by_name_version(model_name, version)
        if not model_info:
            return jsonify({"error": f"Model {model_name} version {version} not found"}), 404
            
        # Get metadata
        metadata = db.get_metadata(model_info['uuid'])
        
        # Construct response
        response = {
            "model_name": model_name,
            "version": version,
            "uuid": model_info['uuid'],
            "filename": f"{model_name}_v{version}.onnx",
            "path": model_info['file_path'],
            "size_bytes": model_info['file_size'],
            "created_at": model_info['created_at'],
            "metadata": metadata or {}
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error getting model details: {e}")
        return jsonify({"error": str(e)}), 500

@model_bp.route('/uuid/<uuid>/detail', methods=['GET'])
def get_model_detail_by_uuid(uuid):
    """
    Get detailed information about a model by UUID.

    Args:
        uuid: UUID of the model
        
    Returns:
        JSON response with detailed model information
    """
    try:
        model_info = db.get_model_by_uuid(uuid)
        if not model_info:
            return jsonify({"error": f"Model with UUID {uuid} not found"}), 404
            
        # Get metadata
        metadata = db.get_metadata(uuid)
        
        # Construct response
        response = {
            "model_name": model_info['name'],
            "version": model_info['version'],
            "uuid": uuid,
            "filename": f"{model_info['name']}_v{model_info['version']}.onnx",
            "path": model_info['file_path'],
            "size_bytes": model_info['file_size'],
            "created_at": model_info['created_at'],
            "metadata": metadata or {}
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error getting model details by UUID: {e}")
        return jsonify({"error": str(e)}), 500
