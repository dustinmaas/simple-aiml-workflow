#!/usr/bin/env python3
"""
Health check routes for the inference server.

This module defines health check endpoints to monitor the server's status.
"""

import time
import logging
from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

# Create Blueprint for health routes
health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    Root health check endpoint.
    
    Returns:
        JSON response with server status
    """
    return jsonify({
        "status": "healthy",
        "server": "inference",
        "timestamp": time.time()
    })
