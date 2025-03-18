#!/usr/bin/env python3
"""
Constants for the inference server.

This module defines constants used throughout the inference server.
"""

import os

# Model server host and port for retrieving models
MODEL_SERVER_HOST = os.environ.get('MODEL_SERVER_HOST', 'model-server')
MODEL_SERVER_PORT = os.environ.get('MODEL_SERVER_PORT', '80')
MODEL_SERVER_URL = f"http://{MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}"

# Inference server host and port
INFERENCE_SERVER_HOST = os.environ.get('INFERENCE_SERVER_HOST', 'inference-server')
INFERENCE_SERVER_PORT = os.environ.get('INFERENCE_SERVER_PORT', '80')
INFERENCE_SERVER_URL = f"http://{INFERENCE_SERVER_HOST}:{INFERENCE_SERVER_PORT}"

# Local cache directory for models
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', '/app/cache/models')

# Maximum content length for requests (10 MB)
MAX_CONTENT_LENGTH = 10 * 1024 * 1024

# Default request timeout for model server requests (in seconds)
REQUEST_TIMEOUT = 30

# Maximum cache size (number of models)
MAX_CACHE_SIZE = int(os.environ.get('MAX_CACHE_SIZE', '10'))
