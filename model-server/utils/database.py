#!/usr/bin/env python3
"""
Database manager for UUID-based model storage.

This module provides a SQLite-based database for storing model information and
metadata with connection management, transaction handling, and error reporting.
"""

import os
import sqlite3
import uuid
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator, ContextManager
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ModelDatabaseError(Exception):
    """Base exception for all database-related errors."""
    pass

class ModelNotFoundError(ModelDatabaseError):
    """Exception raised when a requested model is not found."""
    pass

class ModelExistsError(ModelDatabaseError):
    """Exception raised when attempting to add a model that already exists."""
    pass

class ModelDatabase:
    def __init__(self, db_path: str):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._initialize_db()
    
    @contextmanager
    def _get_connection(self) -> Iterator[Tuple[sqlite3.Connection, sqlite3.Cursor]]:
        """
        Context manager for database connections.
        
        Returns:
            Tuple of (connection, cursor)
            
        Yields:
            Same tuple for use in 'with' statement
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            
            yield conn, cursor
            
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise ModelDatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def _initialize_db(self) -> None:
        """Create database tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with self._get_connection() as (conn, cursor):
            # Create models table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                uuid TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                UNIQUE(name, version)
            )
            ''')
            
            # Create metadata table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                uuid TEXT PRIMARY KEY,
                content JSON NOT NULL,
                FOREIGN KEY (uuid) REFERENCES models(uuid) ON DELETE CASCADE
            )
            ''')
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def add_model(self, name: str, version: str, file_path: str, file_size: int) -> str:
        """
        Add a new model to the database.
        
        Args:
            name: Model name
            version: Model version
            file_path: Path to the model file
            file_size: Size of the model file in bytes
            
        Returns:
            UUID of the new model
            
        Raises:
            ModelExistsError: If a model with the same name and version already exists
        """
        model_uuid = str(uuid.uuid4())
        
        try:
            with self._get_connection() as (conn, cursor):
                try:
                    cursor.execute(
                        "INSERT INTO models (uuid, name, version, file_path, file_size) VALUES (?, ?, ?, ?, ?)",
                        (model_uuid, name, version, file_path, file_size)
                    )
                    conn.commit()
                    logger.info(f"Added model {name} version {version} with UUID {model_uuid}")
                    return model_uuid
                except sqlite3.IntegrityError:
                    # Model with this name and version already exists
                    conn.rollback()
                    logger.warning(f"Model {name} version {version} already exists")
                    raise ModelExistsError(f"Model '{name}' version '{version}' already exists")
        except ModelDatabaseError:
            # Re-raise database errors
            raise
    
    def add_model_with_uuid(self, model_uuid: str, name: str, version: str, file_path: str, file_size: int) -> bool:
        """
        Add a new model to the database with a specific UUID.

        Args:
            model_uuid: UUID to use for the model
            name: Model name
            version: Model version
            file_path: Path to the model file
            file_size: Size of the model file in bytes

        Returns:
            True if the model was added successfully
            
        Raises:
            ModelExistsError: If a model with the same name and version already exists
        """
        try:
            with self._get_connection() as (conn, cursor):
                try:
                    cursor.execute(
                        "INSERT INTO models (uuid, name, version, file_path, file_size) VALUES (?, ?, ?, ?, ?)",
                        (model_uuid, name, version, file_path, file_size)
                    )
                    conn.commit()
                    logger.info(f"Added model {name} version {version} with provided UUID {model_uuid}")
                    return True
                except sqlite3.IntegrityError:
                    # Model with this name and version already exists
                    conn.rollback()
                    logger.warning(f"Model {name} version {version} already exists")
                    raise ModelExistsError(f"Model '{name}' version '{version}' already exists")
        except ModelDatabaseError:
            # Re-raise database errors
            raise
    
    def add_metadata(self, model_uuid: str, metadata_dict: Dict[str, Any]) -> bool:
        """
        Add metadata for a model.
        
        Args:
            model_uuid: UUID of the model
            metadata_dict: Dictionary of metadata
            
        Returns:
            True if successful, False if failed
            
        Raises:
            ModelNotFoundError: If no model with the given UUID exists
        """
        try:
            with self._get_connection() as (conn, cursor):
                try:
                    cursor.execute(
                        "INSERT INTO metadata (uuid, content) VALUES (?, ?)",
                        (model_uuid, json.dumps(metadata_dict))
                    )
                    conn.commit()
                    logger.info(f"Added metadata for model with UUID {model_uuid}")
                    return True
                except sqlite3.IntegrityError:
                    # No model with this UUID or metadata already exists
                    conn.rollback()
                    logger.warning(f"Failed to add metadata for UUID {model_uuid}")
                    raise ModelNotFoundError(f"No model found with UUID '{model_uuid}'")
        except ModelDatabaseError:
            # Re-raise database errors
            raise
    
    def get_model_by_uuid(self, model_uuid: str) -> Dict[str, Any]:
        """
        Get model details by UUID.
        
        Args:
            model_uuid: UUID of the model
            
        Returns:
            Dictionary with model details
            
        Raises:
            ModelNotFoundError: If no model with the given UUID exists
        """
        with self._get_connection() as (conn, cursor):
            cursor.execute("SELECT * FROM models WHERE uuid = ?", (model_uuid,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"No model found with UUID {model_uuid}")
                raise ModelNotFoundError(f"No model found with UUID '{model_uuid}'")
                
            return dict(result)
    
    def get_model_by_name_version(self, name: str, version: str) -> Dict[str, Any]:
        """
        Get model details by name and version.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            Dictionary with model details
            
        Raises:
            ModelNotFoundError: If no model with the given name and version exists
        """
        with self._get_connection() as (conn, cursor):
            cursor.execute("SELECT * FROM models WHERE name = ? AND version = ?", (name, version))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"No model found with name {name} and version {version}")
                raise ModelNotFoundError(f"No model found with name '{name}' and version '{version}'")
                
            return dict(result)
    
    def get_latest_model_by_name(self, name: str) -> Dict[str, Any]:
        """
        Get the latest version of a model by name.
        
        Args:
            name: Model name
            
        Returns:
            Dictionary with model details
            
        Raises:
            ModelNotFoundError: If no model with the given name exists
        """
        with self._get_connection() as (conn, cursor):
            cursor.execute(
                "SELECT * FROM models WHERE name = ? ORDER BY created_at DESC LIMIT 1", 
                (name,)
            )
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"No model found with name {name}")
                raise ModelNotFoundError(f"No model found with name '{name}'")
                
            return dict(result)
    
    def get_metadata(self, model_uuid: str) -> Dict[str, Any]:
        """
        Get metadata for a model.
        
        Args:
            model_uuid: UUID of the model
            
        Returns:
            Dictionary with metadata
            
        Raises:
            ModelNotFoundError: If no metadata found for the model
        """
        with self._get_connection() as (conn, cursor):
            cursor.execute("SELECT content FROM metadata WHERE uuid = ?", (model_uuid,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"No metadata found for model with UUID {model_uuid}")
                raise ModelNotFoundError(f"No metadata found for model with UUID '{model_uuid}'")
                
            return json.loads(result['content'])
    
    def list_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all models grouped by name.
        
        Returns:
            Dictionary with model names as keys and lists of model versions as values
        """
        with self._get_connection() as (conn, cursor):
            cursor.execute("SELECT * FROM models ORDER BY name, version")
            results = cursor.fetchall()
            
            models_by_name = {}
            for row in results:
                model = dict(row)
                name = model['name']
                if name not in models_by_name:
                    models_by_name[name] = []
                models_by_name[name].append(model)
            
            return models_by_name
    
    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a model.
        
        Args:
            name: Model name
            
        Returns:
            List of dictionaries with model details
        """
        with self._get_connection() as (conn, cursor):
            cursor.execute("SELECT * FROM models WHERE name = ? ORDER BY version", (name,))
            results = cursor.fetchall()
            
            if not results:
                logger.info(f"No versions found for model {name}")
                return []
                
            return [dict(row) for row in results]
    
    def delete_model(self, model_uuid: str) -> bool:
        """
        Delete a model and its metadata.
        
        Args:
            model_uuid: UUID of the model
            
        Returns:
            True if model was deleted
            
        Raises:
            ModelNotFoundError: If no model with the given UUID exists
        """
        try:
            with self._get_connection() as (conn, cursor):
                # Due to CASCADE, this will also delete metadata
                cursor.execute("DELETE FROM models WHERE uuid = ?", (model_uuid,))
                
                if cursor.rowcount == 0:
                    logger.warning(f"No model found with UUID {model_uuid}")
                    raise ModelNotFoundError(f"No model found with UUID '{model_uuid}'")
                
                conn.commit()
                logger.info(f"Deleted model with UUID {model_uuid}")
                return True
                
        except ModelDatabaseError:
            # Re-raise database errors
            raise
    
    def delete_model_by_name_version(self, name: str, version: str) -> bool:
        """
        Delete a model by name and version.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if model was deleted
            
        Raises:
            ModelNotFoundError: If no model with the given name and version exists
        """
        try:
            with self._get_connection() as (conn, cursor):
                # Get the UUID first
                cursor.execute("SELECT uuid FROM models WHERE name = ? AND version = ?", (name, version))
                result = cursor.fetchone()
                
                if not result:
                    logger.warning(f"No model found with name {name} and version {version}")
                    raise ModelNotFoundError(f"No model found with name '{name}' and version '{version}'")
                
                model_uuid = result[0]
                
                # Then delete by UUID (which will also delete metadata due to CASCADE)
                cursor.execute("DELETE FROM models WHERE uuid = ?", (model_uuid,))
                conn.commit()
                
                logger.info(f"Deleted model {name} version {version} (UUID: {model_uuid})")
                return True
                
        except ModelDatabaseError:
            # Re-raise database errors
            raise
    
    def delete_all_versions(self, name: str) -> Tuple[int, List[str]]:
        """
        Delete all versions of a model.
        
        Args:
            name: Model name
            
        Returns:
            Tuple of (number of versions deleted, list of errors)
        """
        errors = []
        deleted_count = 0
        
        with self._get_connection() as (conn, cursor):
            # Get all versions first
            cursor.execute("SELECT uuid, version FROM models WHERE name = ?", (name,))
            versions = cursor.fetchall()
            
            if not versions:
                logger.warning(f"No versions found for model {name}")
                return 0, []
            
            # Process each version
            for version in versions:
                model_uuid = version['uuid']
                version_str = version['version']
                
                try:
                    cursor.execute("DELETE FROM models WHERE uuid = ?", (model_uuid,))
                    deleted_count += 1
                    logger.info(f"Deleted model {name} version {version_str} (UUID: {model_uuid})")
                except Exception as e:
                    error_msg = f"Failed to delete {name} version {version_str}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            conn.commit()
            
            return deleted_count, errors
    
    def update_metadata(self, model_uuid: str, metadata_dict: Dict[str, Any]) -> None:
        """
        Update metadata for a model.
        
        Args:
            model_uuid: UUID of the model
            metadata_dict: Dictionary of metadata
            
        Raises:
            ModelNotFoundError: If no model with the given UUID exists
        """
        try:
            with self._get_connection() as (conn, cursor):
                # Check if the model exists
                cursor.execute("SELECT 1 FROM models WHERE uuid = ?", (model_uuid,))
                if not cursor.fetchone():
                    raise ModelNotFoundError(f"No model found with UUID '{model_uuid}'")
                
                # Check if metadata exists
                cursor.execute("SELECT 1 FROM metadata WHERE uuid = ?", (model_uuid,))
                exists = cursor.fetchone() is not None
                
                if exists:
                    cursor.execute(
                        "UPDATE metadata SET content = ? WHERE uuid = ?",
                        (json.dumps(metadata_dict), model_uuid)
                    )
                else:
                    cursor.execute(
                        "INSERT INTO metadata (uuid, content) VALUES (?, ?)",
                        (model_uuid, json.dumps(metadata_dict))
                    )
                    
                conn.commit()
                logger.info(f"Updated metadata for model with UUID {model_uuid}")
                
        except ModelDatabaseError:
            # Re-raise database errors
            raise
            
    def model_exists(self, name: str, version: str) -> bool:
        """
        Check if a model with the given name and version exists.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if the model exists, False otherwise
        """
        with self._get_connection() as (conn, cursor):
            cursor.execute(
                "SELECT 1 FROM models WHERE name = ? AND version = ? LIMIT 1", 
                (name, version)
            )
            return cursor.fetchone() is not None

    def extract_storage_uuid_from_path(self, file_path: str) -> Optional[str]:
        """
        Extract the storage UUID from a file path.
        
        This is useful when the storage UUID might differ from the database UUID.
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Extracted UUID or None if not found
        """
        # Try to extract UUID from the file path
        # Assumes the UUID is the basename of the file (without extension)
        match = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', file_path)
        if match:
            return match.group(1)
        return None
