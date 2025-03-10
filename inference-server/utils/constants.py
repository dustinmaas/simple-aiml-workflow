#!/usr/bin/env python3
"""
Constants for the inference server.

This module defines constants used throughout the inference server.
"""

import os

# Model server URL for retrieving models
MODEL_SERVER_URL = os.environ.get('MODEL_SERVER_URL', 'http://model-server:5000')

# Local cache directory for models
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', '/app/cache/models')

# Maximum content length for requests (10 MB)
MAX_CONTENT_LENGTH = 10 * 1024 * 1024

# Default request timeout for model server requests (in seconds)
REQUEST_TIMEOUT = 30

# Maximum cache size (number of models)
MAX_CACHE_SIZE = int(os.environ.get('MAX_CACHE_SIZE', '10'))
