#!/usr/bin/env python3
"""
Application factory for the model server.

This module provides a function to create and configure the Flask application.
"""

import os
import logging
import sys
from flask import Flask

# Add the current directory to the path so we can use absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from routes.health import health_bp
from routes.model_routes import model_bp
from utils.constants import MODEL_DB_PATH, MODEL_STORAGE_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(test_config=None):
    """
    Create and configure the Flask application.
    
    Args:
        test_config: Configuration to use for testing (optional)
        
    Returns:
        Configured Flask application
    """
    # Create and configure the app
    app = Flask(__name__)
    
    # Load default configuration
    app.config.from_mapping(
        MODELS_DIR='/app/models',
        MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50 MB max upload size
    )
    
    # Override with test config if provided
    if test_config:
        app.config.update(test_config)
    
    # Ensure the directories exist
    # Make sure storage directory for model files exists
    os.makedirs(MODEL_STORAGE_DIR, exist_ok=True)
    # Make sure database directory exists
    os.makedirs(os.path.dirname(MODEL_DB_PATH), exist_ok=True)
    
    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(model_bp)
    
    # Apply custom error handlers
    register_error_handlers(app)
    
    # Log that the app was initialized
    logger.info(f"Initialized model server with UUID storage at: {MODEL_STORAGE_DIR}")
    logger.info(f"Model database at: {MODEL_DB_PATH}")
    
    return app

def register_error_handlers(app):
    """
    Register custom error handlers with the Flask application.
    
    Args:
        app: Flask application
    """
    @app.errorhandler(404)
    def not_found(error):
        return {
            "error": "Not found",
            "message": "The requested resource was not found."
        }, 404
    
    @app.errorhandler(400)
    def bad_request(error):
        return {
            "error": "Bad request",
            "message": str(error)
        }, 400
    
    @app.errorhandler(500)
    def server_error(error):
        logger.error(f"Server error: {error}")
        return {
            "error": "Internal server error",
            "message": "An unexpected error occurred."
        }, 500
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return {
            "error": "Payload too large",
            "message": f"The request payload exceeds the maximum allowed size ({app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024):.1f} MB)."
        }, 413
