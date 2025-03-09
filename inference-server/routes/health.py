#!/usr/bin/env python3
"""
Health check routes for the inference server.

This module defines routes for checking the health and status of the server.
"""

import os
import logging
import requests
from flask import Blueprint, jsonify, current_app

from utils.constants import MODEL_SERVER_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

# Create Blueprint for health routes
health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON response indicating server status
    """
    return jsonify({"status": "healthy"}), 200

@health_bp.route('/status', methods=['GET'])
def server_status():
    """
    More detailed server status endpoint.
    
    Returns:
        JSON response with server status details including model server connectivity
    """
    model_server_url = current_app.config['MODEL_SERVER_URL']
    
    status = {
        "status": "healthy",
        "model_server": {
            "url": model_server_url,
            "connection": "unknown"
        },
        "cache": {
            "models_cached": len(current_app.model_cache.cache)
        }
    }
    
    # Check model server connection
    try:
        response = requests.get(f"{model_server_url}/health", timeout=MODEL_SERVER_REQUEST_TIMEOUT)
        if response.status_code == 200:
            status["model_server"]["connection"] = "connected"
            status["model_server"]["status"] = response.json().get("status", "unknown")
        else:
            status["model_server"]["connection"] = "error"
            status["model_server"]["error"] = f"Status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        status["model_server"]["connection"] = "error"
        status["model_server"]["error"] = str(e)
        
        # If we can't connect to the model server, the inference server is not fully healthy
        status["status"] = "degraded"
    
    return jsonify(status)

@health_bp.route('/cache', methods=['GET'])
def cache_info():
    """
    Get information about the model cache.
    
    Returns:
        JSON response with cache information
    """
    cache_info = current_app.model_cache.get_cache_info()
    
    return jsonify({
        "cached_models": len(cache_info),
        "models": cache_info
    })

@health_bp.route('/cache/clear', methods=['POST'])
def clear_cache():
    """
    Clear the model cache.
    
    Returns:
        JSON response indicating success
    """
    current_app.model_cache.clear()
    
    return jsonify({
        "success": True,
        "message": "Model cache cleared"
    })
