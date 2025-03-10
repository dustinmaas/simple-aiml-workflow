#!/usr/bin/env python3
"""
Storage manager for UUID-based model files.

This module provides utilities for storing and retrieving model files using UUIDs.
"""

import os
import shutil
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

class ModelStorage:
    def __init__(self, storage_dir: str):
        """
        Initialize the storage manager.
        
        Args:
            storage_dir: Directory where model files will be stored
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"Storage initialized at {storage_dir}")
    
    def _get_model_path(self, model_uuid: str) -> str:
        """
        Get the file path for a model UUID.
        
        Args:
            model_uuid: UUID of the model
            
        Returns:
            Absolute path to the model file
        """
        return os.path.join(self.storage_dir, f"{model_uuid}.onnx")
    
    def store_model(self, model_uuid: str, model_data: bytes) -> Tuple[str, int]:
        """
        Store a model file with its UUID as the filename.
        
        Args:
            model_uuid: UUID of the model
            model_data: Binary content of the model file
            
        Returns:
            Tuple of (file path, file size in bytes)
        """
        file_path = self._get_model_path(model_uuid)
        
        with open(file_path, 'wb') as f:
            f.write(model_data)
        
        file_size = len(model_data)
        logger.info(f"Stored model with UUID {model_uuid}, size: {file_size} bytes")
        
        return file_path, file_size
    
    def get_model(self, model_uuid: str) -> Optional[bytes]:
        """
        Get a model file by UUID.
        
        Args:
            model_uuid: UUID of the model
            
        Returns:
            Binary content of the model file or None if not found
        """
        file_path = self._get_model_path(model_uuid)
        
        if not os.path.exists(file_path):
            logger.warning(f"Model file not found for UUID {model_uuid}")
            return None
        
        with open(file_path, 'rb') as f:
            model_data = f.read()
            
        logger.info(f"Retrieved model with UUID {model_uuid}, size: {len(model_data)} bytes")
        return model_data
    
    def delete_model(self, model_uuid: str) -> bool:
        """
        Delete a model file by UUID.
        
        Args:
            model_uuid: UUID of the model
            
        Returns:
            True if the file was deleted, False if it didn't exist
        """
        file_path = self._get_model_path(model_uuid)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted model file for UUID {model_uuid}")
            return True
        
        logger.warning(f"No model file found to delete for UUID {model_uuid}")
        return False
    
    def get_model_size(self, model_uuid: str) -> Optional[int]:
        """
        Get the size of a model file.
        
        Args:
            model_uuid: UUID of the model
            
        Returns:
            Size of the model file in bytes or None if not found
        """
        file_path = self._get_model_path(model_uuid)
        
        if not os.path.exists(file_path):
            return None
        
        return os.path.getsize(file_path)
    
    def model_exists(self, model_uuid: str) -> bool:
        """
        Check if a model file exists.
        
        Args:
            model_uuid: UUID of the model
            
        Returns:
            True if the file exists, False otherwise
        """
        file_path = self._get_model_path(model_uuid)
        return os.path.exists(file_path)
    
    def copy_model(self, source_uuid: str, target_uuid: str) -> bool:
        """
        Copy a model file from one UUID to another.
        
        Args:
            source_uuid: UUID of the source model
            target_uuid: UUID of the target model
            
        Returns:
            True if successful, False if the source doesn't exist
        """
        source_path = self._get_model_path(source_uuid)
        target_path = self._get_model_path(target_uuid)
        
        if not os.path.exists(source_path):
            logger.warning(f"Source model with UUID {source_uuid} not found for copying")
            return False
        
        shutil.copy2(source_path, target_path)
        logger.info(f"Copied model from UUID {source_uuid} to UUID {target_uuid}")
        
        return True
