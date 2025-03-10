#!/usr/bin/env python3
"""
Model service for the inference server.

This module provides utilities for retrieving models from the model server
and handling the UUID extraction pattern for model retrieval.
"""

import os
import logging
import requests
import json
import shutil
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict

from utils.constants import (
    MODEL_SERVER_URL,
    MODEL_CACHE_DIR,
    REQUEST_TIMEOUT,
    MAX_CACHE_SIZE
)

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, cache_dir: str = MODEL_CACHE_DIR):
        """
        Initialize the model service.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.model_server_url = MODEL_SERVER_URL
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Model service initialized with cache at {cache_dir}")
        
        # Simple LRU cache to keep track of cached models
        self.model_cache = OrderedDict()
    
    def get_model_by_uuid(self, uuid: str) -> Tuple[Optional[str], Optional[Dict[Any, Any]]]:
        """
        Get a model by its UUID.
        
        Args:
            uuid: UUID of the model
            
        Returns:
            Tuple of (file path to the cached model, model metadata)
        """
        # Check cache first
        if uuid in self.model_cache:
            self.model_cache.move_to_end(uuid)
            logger.info(f"Model {uuid} found in cache")
            cache_path = os.path.join(self.cache_dir, f"{uuid}.onnx")
            if os.path.exists(cache_path):
                return cache_path, self._get_metadata_by_uuid(uuid)
        
        # Download model from model server
        try:
            logger.info(f"Downloading model with UUID {uuid} from model server")
            model_url = f"{self.model_server_url}/models/uuid/{uuid}"
            response = requests.get(model_url, timeout=REQUEST_TIMEOUT, stream=True)
            
            if response.status_code != 200:
                logger.error(f"Failed to get model with UUID {uuid}: {response.text}")
                return None, None
            
            # Extract storage UUID from the filename if provided in Content-Disposition header
            storage_uuid = uuid  # Default to database UUID
            content_disposition = response.headers.get('Content-Disposition', '')
            
            if 'filename=' in content_disposition:
                try:
                    # Extract filename from Content-Disposition header
                    filename = content_disposition.split('filename=')[1].strip('"\'')
                    # The model file path in the database should contain the storage UUID
                    # Extract UUID from model detail to get the storage UUID
                    detail_response = requests.get(
                        f"{self.model_server_url}/models/uuid/{uuid}/detail",
                        timeout=REQUEST_TIMEOUT
                    )
                    if detail_response.status_code == 200:
                        detail = detail_response.json()
                        file_path = detail.get('path', '')
                        if file_path:
                            # Extract storage UUID from file path
                            # Format of file_path is: "/data/models/[storage_uuid].onnx"
                            try:
                                storage_uuid = os.path.basename(file_path).replace(".onnx", "")
                                logger.info(f"Database UUID: {uuid}, Storage UUID: {storage_uuid}")
                            except Exception as e:
                                logger.error(f"Failed to extract storage UUID from path {file_path}: {e}")
                                # Fallback to database UUID
                                storage_uuid = uuid
                except Exception as e:
                    logger.warning(f"Could not extract storage UUID from headers: {e}")
            
            # Save model to cache with storage UUID
            cache_path = os.path.join(self.cache_dir, f"{storage_uuid}.onnx")
            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Update cache
            self.model_cache[uuid] = storage_uuid
            if len(self.model_cache) > MAX_CACHE_SIZE:
                # Remove oldest item from cache
                oldest_uuid, oldest_storage_uuid = self.model_cache.popitem(last=False)
                oldest_path = os.path.join(self.cache_dir, f"{oldest_storage_uuid}.onnx")
                if os.path.exists(oldest_path):
                    os.remove(oldest_path)
                    logger.info(f"Removed oldest model {oldest_uuid} from cache")
            
            return cache_path, self._get_metadata_by_uuid(uuid)
            
        except Exception as e:
            logger.error(f"Error retrieving model with UUID {uuid}: {e}")
            return None, None
    
    def get_model_by_name_version(self, name: str, version: str) -> Tuple[Optional[str], Optional[Dict[Any, Any]]]:
        """
        Get a model by its name and version.
        
        Args:
            name: Name of the model
            version: Version of the model
            
        Returns:
            Tuple of (file path to the cached model, model metadata)
        """
        try:
            # First, get the model's UUID from the model server
            logger.info(f"Getting UUID for model {name} version {version}")
            detail_url = f"{self.model_server_url}/models/{name}/versions/{version}/detail"
            response = requests.get(detail_url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code != 200:
                logger.error(f"Failed to get details for model {name} version {version}: {response.text}")
                return None, None
            
            model_detail = response.json()
            uuid = model_detail.get('uuid')
            
            if not uuid:
                logger.error(f"No UUID found for model {name} version {version}")
                return None, None
            
            # Now get the model by UUID
            return self.get_model_by_uuid(uuid)
            
        except Exception as e:
            logger.error(f"Error retrieving model {name} version {version}: {e}")
            return None, None
    
    def get_latest_model_version(self, name: str) -> Tuple[Optional[str], Optional[Dict[Any, Any]]]:
        """
        Get the latest version of a model.
        
        Args:
            name: Name of the model
            
        Returns:
            Tuple of (file path to the cached model, model metadata)
        """
        try:
            # Get the model versions from the model server
            logger.info(f"Getting latest version for model {name}")
            versions_url = f"{self.model_server_url}/models/{name}/versions"
            response = requests.get(versions_url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code != 200:
                logger.error(f"Failed to get versions for model {name}: {response.text}")
                return None, None
            
            versions_data = response.json()
            versions = versions_data.get('versions', [])
            
            if not versions:
                logger.error(f"No versions found for model {name}")
                return None, None
            
            # Sort versions by creation time (newest first)
            versions.sort(key=lambda x: x['created'], reverse=True)
            latest_version = versions[0]
            version = latest_version.get('version')
            
            # Get the model by name and version
            return self.get_model_by_name_version(name, version)
            
        except Exception as e:
            logger.error(f"Error retrieving latest version for model {name}: {e}")
            return None, None
    
    def _get_metadata_by_uuid(self, uuid: str) -> Optional[Dict[Any, Any]]:
        """
        Get metadata for a model by UUID.
        
        Args:
            uuid: UUID of the model
            
        Returns:
            Model metadata dictionary or None if not found
        """
        try:
            metadata_url = f"{self.model_server_url}/models/uuid/{uuid}/metadata"
            response = requests.get(metadata_url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code != 200:
                logger.warning(f"Failed to get metadata for model {uuid}: {response.text}")
                return None
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error retrieving metadata for model {uuid}: {e}")
            return None
    
    def clear_cache(self) -> None:
        """
        Clear the model cache.
        """
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            self.model_cache.clear()
            logger.info("Model cache cleared")
            
        except Exception as e:
            logger.error(f"Error clearing model cache: {e}")
