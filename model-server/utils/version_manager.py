#!/usr/bin/env python3
"""
Model version management utilities.

This module provides functions for parsing, validating, and managing model versions.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Any

from utils.constants import MODELS_DIR, VERSION_PATTERN_STR

logger = logging.getLogger(__name__)

# Version regex pattern for model filenames: name_v1.0.0.onnx
VERSION_PATTERN = re.compile(VERSION_PATTERN_STR)

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

def get_model_versions(model_name: str, models_dir: str) -> List[Tuple[str, Tuple[int, int, int]]]:
    """
    Get all versions of a specific model.
    
    Args:
        model_name: Base name of the model
        models_dir: Directory containing model files
        
    Returns:
        List of tuples (filename, (major, minor, patch)) sorted by version
    """
    versions = []
    
    for filename in os.listdir(models_dir):
        parsed = parse_model_filename(filename)
        if parsed and parsed[0] == model_name:
            versions.append((filename, parsed[1]))
    
    # Sort by version (major, minor, patch)
    return sorted(versions, key=lambda x: x[1])

def get_latest_version(model_name: str, models_dir: str) -> Optional[str]:
    """
    Get the latest version of a model.
    
    Args:
        model_name: Base name of the model
        models_dir: Directory containing model files
        
    Returns:
        Filename of the latest version or None if no versions found
    """
    versions = get_model_versions(model_name, models_dir)
    if not versions:
        return None
    
    # Return the filename of the latest version
    return versions[-1][0]

def version_to_string(version: Tuple[int, int, int]) -> str:
    """Convert version tuple to string."""
    return f"{version[0]}.{version[1]}.{version[2]}"

def group_models_by_name(models_dir: str) -> Dict[str, List[Dict]]:
    """
    Group all models by base name with their versions.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Dictionary mapping model names to lists of version information
    """
    result = {}
    
    for filename in os.listdir(models_dir):
        parsed = parse_model_filename(filename)
        if parsed:
            model_name, version = parsed
            
            if model_name not in result:
                result[model_name] = []
                
            version_str = version_to_string(version)
            result[model_name].append({
                "version": version_str,
                "filename": filename,
                "path": os.path.join(models_dir, filename)
            })
    
    # Sort versions within each model
    for model_name in result:
        result[model_name].sort(key=lambda x: tuple(map(int, x["version"].split("."))))
    
    return result

def get_next_version(model_name: str, models_dir: str) -> Tuple[int, int, int]:
    """
    Calculate the next version number for a model.
    
    Args:
        model_name: Base name of the model
        models_dir: Directory containing model files
        
    Returns:
        Tuple of (major, minor, patch) for the next version
    """
    versions = get_model_versions(model_name, models_dir)
    
    if not versions:
        # No versions exist, start with 1.0.0
        return (1, 0, 0)
    
    # Get the latest version and increment the patch version
    _, latest_version = versions[-1]
    return (latest_version[0], latest_version[1], latest_version[2] + 1)
