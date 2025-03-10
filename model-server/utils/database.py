#!/usr/bin/env python3
"""
Database manager for UUID-based model storage.

This module provides a SQLite-based database for storing model information and metadata.
"""

import os
import sqlite3
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class ModelDatabase:
    def __init__(self, db_path: str):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Create tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
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
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def add_model(self, name: str, version: str, file_path: str, file_size: int) -> Optional[str]:
        """
        Add a new model to the database.
        
        Args:
            name: Model name
            version: Model version
            file_path: Path to the model file
            file_size: Size of the model file in bytes
            
        Returns:
            UUID of the new model or None if model already exists
        """
        model_uuid = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
            return None
        finally:
            conn.close()
    
    def add_metadata(self, model_uuid: str, metadata_dict: Dict[str, Any]) -> bool:
        """
        Add metadata for a model.
        
        Args:
            model_uuid: UUID of the model
            metadata_dict: Dictionary of metadata
            
        Returns:
            True if successful, False if failed
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
            return False
        finally:
            conn.close()
    
    def get_model_by_uuid(self, model_uuid: str) -> Optional[Dict[str, Any]]:
        """
        Get model details by UUID.
        
        Args:
            model_uuid: UUID of the model
            
        Returns:
            Dictionary with model details or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM models WHERE uuid = ?", (model_uuid,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return dict(result)
        return None
    
    def get_model_by_name_version(self, name: str, version: str) -> Optional[Dict[str, Any]]:
        """
        Get model details by name and version.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            Dictionary with model details or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM models WHERE name = ? AND version = ?", (name, version))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return dict(result)
        return None
    
    def get_latest_model_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a model by name.
        
        Args:
            name: Model name
            
        Returns:
            Dictionary with model details or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM models WHERE name = ? ORDER BY created_at DESC LIMIT 1", 
            (name,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return dict(result)
        return None
    
    def get_metadata(self, model_uuid: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a model.
        
        Args:
            model_uuid: UUID of the model
            
        Returns:
            Dictionary with metadata or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT content FROM metadata WHERE uuid = ?", (model_uuid,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result['content'])
        return None
    
    def list_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all models grouped by name.
        
        Returns:
            Dictionary with model names as keys and lists of model versions as values
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM models ORDER BY name, version")
        results = cursor.fetchall()
        conn.close()
        
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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM models WHERE name = ? ORDER BY version", (name,))
        results = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in results]
    
    def delete_model(self, model_uuid: str) -> bool:
        """
        Delete a model and its metadata.
        
        Args:
            model_uuid: UUID of the model
            
        Returns:
            True if model was deleted, False if model was not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Due to CASCADE, this will also delete metadata
            cursor.execute("DELETE FROM models WHERE uuid = ?", (model_uuid,))
            deleted = cursor.rowcount > 0
            conn.commit()
            
            if deleted:
                logger.info(f"Deleted model with UUID {model_uuid}")
            else:
                logger.warning(f"No model found with UUID {model_uuid}")
                
            return deleted
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting model with UUID {model_uuid}: {e}")
            return False
        finally:
            conn.close()
    
    def delete_model_by_name_version(self, name: str, version: str) -> bool:
        """
        Delete a model by name and version.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if model was deleted, False if model was not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get the UUID first
            cursor.execute("SELECT uuid FROM models WHERE name = ? AND version = ?", (name, version))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"No model found with name {name} and version {version}")
                return False
            
            model_uuid = result[0]
            
            # Then delete by UUID (which will also delete metadata due to CASCADE)
            cursor.execute("DELETE FROM models WHERE uuid = ?", (model_uuid,))
            conn.commit()
            
            logger.info(f"Deleted model {name} version {version} (UUID: {model_uuid})")
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting model {name} version {version}: {e}")
            return False
        finally:
            conn.close()
    
    def delete_all_versions(self, name: str) -> Tuple[int, List[str]]:
        """
        Delete all versions of a model.
        
        Args:
            name: Model name
            
        Returns:
            Tuple of (number of versions deleted, list of errors)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all versions first
        cursor.execute("SELECT uuid, version FROM models WHERE name = ?", (name,))
        versions = cursor.fetchall()
        
        if not versions:
            logger.warning(f"No versions found for model {name}")
            return 0, []
        
        deleted_count = 0
        errors = []
        
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
        conn.close()
        
        return deleted_count, errors
    
    def update_metadata(self, model_uuid: str, metadata_dict: Dict[str, Any]) -> bool:
        """
        Update metadata for a model.
        
        Args:
            model_uuid: UUID of the model
            metadata_dict: Dictionary of metadata
            
        Returns:
            True if successful, False if failed
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
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
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating metadata for UUID {model_uuid}: {e}")
            return False
        finally:
            conn.close()
            
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
            True if the model was added successfully, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

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
            return False
        except Exception as e:
            conn.rollback()
            logger.error(f"Error adding model to database: {e}")
            return False
        finally:
            conn.close()
