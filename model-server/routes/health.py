#!/usr/bin/env python3
"""
Health check routes for the model server.

This module defines routes for checking the health and status of the server.
"""

import os
import logging
from flask import Blueprint, jsonify, current_app

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
    return jsonify({"status": "ok"})

@health_bp.route('/status', methods=['GET'])
def server_status():
    """
    More detailed server status endpoint.
    
    Returns:
        JSON response with server status details
    """
    models_dir = os.environ.get('MODELS_DIR', '/app/models')
    
    try:
        disk_free = os.statvfs(models_dir).f_bavail * os.statvfs(models_dir).f_frsize
        disk_total = os.statvfs(models_dir).f_blocks * os.statvfs(models_dir).f_frsize
        disk_used = disk_total - disk_free
        
        status = {
            "status": "ok",
            "storage": {
                "models_dir": models_dir,
                "total_bytes": disk_total,
                "used_bytes": disk_used,
                "free_bytes": disk_free,
                "usage_percent": (disk_used / disk_total) * 100 if disk_total > 0 else 0
            },
            "models_dir_exists": os.path.isdir(models_dir),
            "models_dir_writable": os.access(models_dir, os.W_OK)
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting server status: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error getting server status: {str(e)}"
        }), 500
