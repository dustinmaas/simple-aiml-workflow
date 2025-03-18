#!/usr/bin/env python3
"""
Constants for the model server.

This module defines constants used throughout the model server.
"""

import os

# Path to the database file
MODEL_DB_PATH = os.environ.get('MODEL_DB_PATH', '/data/db/models.db')

# Path to the model storage directory
MODEL_STORAGE_DIR = os.environ.get('MODEL_STORAGE_DIR', '/data/models')

# Maximum content length for uploads (50 MB)
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

# Model Server Host and Port
MODEL_SERVER_HOST = os.environ.get('MODEL_SERVER_HOST', 'localhost')
MODEL_SERVER_PORT = os.environ.get('MODEL_SERVER_PORT', '80')

# Model Server URL (constructed from host and port)
MODEL_SERVER_URL = f"http://{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}"
