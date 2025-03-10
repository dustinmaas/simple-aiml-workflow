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

# Model Server URL (default when running inside the container)
MODEL_SERVER_URL = os.environ.get('MODEL_SERVER_URL', 'http://localhost:5000')
