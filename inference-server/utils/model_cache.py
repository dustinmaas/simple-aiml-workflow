#!/usr/bin/env python3
"""
Model caching utilities for the inference server.

This module provides a cache for storing loaded ONNX models in memory.
"""

import os
import logging
import tempfile
import requests
import onnxruntime as ort
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from utils.constants import MODEL_SERVER_REQUEST_TIMEOUT, MAX_CACHE_SIZE, CACHE_EXPIRY_SECONDS

logger = logging.getLogger(__name__)

class ModelCache:
    """
    Cache for storing loaded ONNX models in memory.
    
    This cache stores ONNX Runtime InferenceSession objects along with their metadata,
    keyed by a combination of model name and version.
    """
    
    def __init__(self, model_server_url: str):
        """
        Initialize the model cache.
        
        Args:
            model_server_url: URL of the model server to fetch models from
        """
        self.model_server_url = model_server_url
        self.cache: Dict[str, Tuple[ort.InferenceSession, Dict[str, Any]]] = {}
        logger.info(f"Initialized model cache with model server: {model_server_url}")
    
    def get_cache_key(self, model_name: str, version: Optional[str] = None) -> str:
        """
        Get a cache key for the specified model and version.
        
        Args:
            model_name: Base name of the model
            version: Specific version string, or None for latest
            
        Returns:
            Cache key string
        """
        return f"{model_name}:{version or 'latest'}"
    
    def get_model(self, model_name: str, version: Optional[str] = None) -> Tuple[ort.InferenceSession, Dict[str, Any]]:
        """
        Get a model from cache or fetch it from the model server.
        
        Args:
            model_name: Base name of the model
            version: Specific version to retrieve, or None for latest
            
        Returns:
            Tuple of (ONNX Runtime InferenceSession, metadata dict)
            
        Raises:
            Exception: If the model cannot be fetched or loaded
        """
        cache_key = self.get_cache_key(model_name, version)
        
        # Check if model is in cache
        if cache_key in self.cache:
            logger.info(f"Using cached model {cache_key}")
            return self.cache[cache_key]
        
        # Model not in cache, fetch from server
        logger.info(f"Model {cache_key} not in cache, fetching from server")
        session, metadata = self._fetch_model(model_name, version)
        
        # Add to cache
        self.cache[cache_key] = (session, metadata)
        return self.cache[cache_key]
    
    def _fetch_model(self, model_name: str, version: Optional[str] = None) -> Tuple[ort.InferenceSession, Dict[str, Any]]:
        """
        Fetch a model from the model server.
        
        Args:
            model_name: Base name of the model
            version: Specific version to retrieve, or None for latest
            
        Returns:
            Tuple of (ONNX Runtime InferenceSession, metadata dict)
            
        Raises:
            Exception: If the model cannot be fetched or loaded
        """
        try:
            # Determine the URL to fetch from
            if version == 'latest' or version is None:
                url = f"{self.model_server_url}/models/{model_name}/versions/latest"
            else:
                url = f"{self.model_server_url}/models/{model_name}/versions/{version}"
            
            logger.info(f"Fetching model from {url}")
            response = requests.get(url, timeout=MODEL_SERVER_REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Create a temporary file for the model
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as temp_file:
                temp_model_path = temp_file.name
                temp_file.write(response.content)
            
            try:
                # Load the ONNX model
                session = ort.InferenceSession(temp_model_path)
                
                # Fetch metadata from separate endpoint
                metadata = self._fetch_metadata(model_name, version)
                
                # If no metadata from separate endpoint, try embedded metadata
                if not metadata:
                    metadata = self._extract_embedded_metadata(session)
                
                return session, metadata
                
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching model {model_name} version {version}: {e}")
            raise Exception(f"Error fetching model: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise Exception(f"Error loading model: {str(e)}")
    
    def _fetch_metadata(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch metadata for a model from the model server.
        
        Args:
            model_name: Base name of the model
            version: Specific version to retrieve, or None for latest
            
        Returns:
            Dictionary containing model metadata or empty dict if metadata can't be fetched
        """
        try:
            version_str = version if version and version != 'latest' else 'latest'
            url = f"{self.model_server_url}/models/{model_name}/versions/{version_str}/metadata"
            
            logger.info(f"Fetching metadata from {url}")
            response = requests.get(url, timeout=MODEL_SERVER_REQUEST_TIMEOUT)
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching metadata for {model_name} version {version}: {e}")
            return {}  # Return empty dict if metadata can't be fetched
        except Exception as e:
            logger.warning(f"Error processing metadata: {e}")
            return {}
    
    def _extract_embedded_metadata(self, session: ort.InferenceSession) -> Dict[str, Any]:
        """
        Extract embedded metadata from an ONNX model.
        
        Args:
            session: ONNX Runtime InferenceSession
            
        Returns:
            Dictionary containing model metadata or empty dict if no metadata found
        """
        try:
            embedded_metadata = session.get_modelmeta().custom_metadata_map
            if embedded_metadata:
                metadata = {k: v for k, v in embedded_metadata.items()}
                logger.info(f"Using embedded metadata: {metadata}")
                return metadata
        except Exception as e:
            logger.warning(f"Failed to extract embedded metadata: {e}")
        
        return {}
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()
        logger.info("Model cache cleared")
    
    def remove(self, model_name: str, version: Optional[str] = None) -> bool:
        """
        Remove a specific model from the cache.
        
        Args:
            model_name: Base name of the model
            version: Specific version to remove, or None for latest
            
        Returns:
            True if the model was in the cache and removed, False otherwise
        """
        cache_key = self.get_cache_key(model_name, version)
        if cache_key in self.cache:
            del self.cache[cache_key]
            logger.info(f"Removed model {cache_key} from cache")
            return True
        return False
    
    def get_cache_info(self) -> List[Dict[str, Any]]:
        """
        Get information about cached models.
        
        Returns:
            List of dictionaries containing information about each cached model
        """
        info = []
        for key, (session, metadata) in self.cache.items():
            model_name, version = key.split(':')
            model_info = {
                "model_name": model_name,
                "version": version,
                "providers": session.get_providers(),
                "inputs": [{
                    "name": input.name,
                    "shape": input.shape,
                    "type": input.type
                } for input in session.get_inputs()],
                "outputs": [{
                    "name": output.name,
                    "shape": output.shape,
                    "type": output.type
                } for output in session.get_outputs()],
                "metadata": metadata
            }
            info.append(model_info)
        return info
