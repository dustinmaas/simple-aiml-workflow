#!/usr/bin/env python3
"""
Application factory for the inference server.

This module provides a function to create and configure the Flask application.
"""

import os
import logging
import sys
from flask import Flask

# Add the current directory to the path so we can use absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from routes.health import health_bp
from routes.inference_routes import inference_bp
from utils.constants import MODEL_CACHE_DIR, MODEL_SERVER_URL
from utils.model_service import ModelService

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
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,  # 10 MB max request size
        MODEL_SERVER_URL=MODEL_SERVER_URL,
    )
    
    # Override with test config if provided
    if test_config:
        app.config.update(test_config)
    
    # Ensure the model cache directory exists
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    # Initialize model service and attach to app
    # Note: ModelService handles file-based model caching
    app.model_service = ModelService(MODEL_CACHE_DIR)
    
    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(inference_bp)
    
    # Apply custom error handlers
    register_error_handlers(app)
    
    # Log that the app was initialized
    logger.info(f"Initialized inference server with model cache at: {MODEL_CACHE_DIR}")
    logger.info(f"Using model server at: {MODEL_SERVER_URL}")
    
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
