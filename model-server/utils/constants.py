#!/usr/bin/env python3
"""
Constants for the model server.

This module defines constants used throughout the model server.
"""

import os

# Path to the models directory
MODELS_DIR = '/app/models'

# Maximum content length for uploads (50 MB)
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

# Default version pattern for model filenames: name_v1.0.0.onnx
VERSION_PATTERN_STR = r'^(.+)_v(\d+)\.(\d+)\.(\d+)\.onnx$'
