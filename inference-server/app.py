#!/usr/bin/env python3
"""
Inference Server API for ONNX models.

This API provides inference endpoints for ONNX models retrieved from the model server.
It implements the UUID extraction pattern to handle potential UUID mismatches between
database UUIDs and storage UUIDs.
"""

import os
import sys
import logging
from app_factory import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can use absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create the application
app = create_app()

if __name__ == '__main__':
    # Get configuration from environment
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting inference server on {host}:{port} (debug={debug_mode})")
    app.run(host=host, port=port, debug=debug_mode)
