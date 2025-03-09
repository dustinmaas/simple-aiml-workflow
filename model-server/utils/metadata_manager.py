#!/usr/bin/env python3
"""
Model metadata management utilities.

This module provides functions for saving, loading, and validating model metadata.
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_metadata_path(model_path: str) -> str:
    """
    Get the path to the metadata file for a given model path.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Path to the corresponding metadata file
    """
    return model_path.replace('.onnx', '.metadata.json')

def save_metadata(model_path: str, metadata: Dict[str, Any]) -> None:
    """
    Save metadata to a JSON file alongside the model.
    
    Args:
        model_path: Path to the model file
        metadata: Dictionary containing model metadata
    """
    metadata_path = get_metadata_path(model_path)
    
    # Add timestamp if not present
    if 'upload_time' not in metadata:
        metadata['upload_time'] = datetime.datetime.now().isoformat()
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_path}")

def load_metadata(model_path: str) -> Dict[str, Any]:
    """
    Load metadata from a JSON file if it exists.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary containing model metadata or empty dict if no metadata file exists
    """
    metadata_path = get_metadata_path(model_path)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding metadata from {metadata_path}: {e}")
            return {}
    return {}

def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Validate model metadata schema.
    
    Args:
        metadata: Dictionary containing model metadata
        
    Returns:
        True if metadata is valid, False otherwise
    """
    # Basic validation for required fields
    required_fields = ['model_name', 'version']
    
    for field in required_fields:
        if field not in metadata:
            logger.warning(f"Missing required field in metadata: {field}")
            return False
    
    return True

def clean_metadata(model_path: str) -> bool:
    """
    Delete metadata file if it exists.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        True if metadata was deleted or didn't exist, False if deletion failed
    """
    metadata_path = get_metadata_path(model_path)
    if os.path.exists(metadata_path):
        try:
            os.remove(metadata_path)
            logger.info(f"Deleted metadata file: {metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting metadata file {metadata_path}: {e}")
            return False
    return True

def merge_metadata(existing: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge new metadata into existing metadata.
    
    Args:
        existing: Existing metadata dictionary
        new_data: New metadata to merge in
        
    Returns:
        Merged metadata dictionary
    """
    result = existing.copy()
    
    for key, value in new_data.items():
        # Don't overwrite existing timestamps unless explicitly provided
        if key == 'upload_time' and key in result and 'force_timestamp' not in new_data:
            continue
        
        result[key] = value
    
    # Remove control fields not meant to be stored
    if 'force_timestamp' in result:
        del result['force_timestamp']
    
    return result

def enriched_metadata_response(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare metadata for API response with additional information.
    
    Args:
        metadata: Raw metadata dictionary
        
    Returns:
        Enhanced metadata for API response
    """
    response = metadata.copy()
    
    # Add upload time in readable format if it exists
    if 'upload_time' in response:
        try:
            timestamp = datetime.datetime.fromisoformat(response['upload_time'])
            response['upload_time_formatted'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            # Keep original if we can't parse it
            pass
            
    return response
