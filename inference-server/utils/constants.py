#!/usr/bin/env python3
"""
Constants for the inference server.

This module defines constants used throughout the inference server.
"""

import os

# Default model server URL
MODEL_SERVER_URL = os.environ.get('MODEL_SERVER_URL', 'http://model-server:5000')

# Default model name to use when none is specified
DEFAULT_MODEL_NAME = 'linear_regression_model'

# Default version to use when none is specified
DEFAULT_MODEL_VERSION = 'latest'

# Maximum content length for uploads (50 MB)
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

# Request timeout for model server communication (in seconds)
MODEL_SERVER_REQUEST_TIMEOUT = 30

# Cache configuration
MAX_CACHE_SIZE = 10  # Maximum number of models to keep in cache
CACHE_EXPIRY_SECONDS = 3600  # Time in seconds after which a cached model is refreshed
