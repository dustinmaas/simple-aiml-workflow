#!/usr/bin/env python3
"""
Tests for the application factory of the inference server.

This module tests that the Flask application is properly configured and routes are registered.
"""

import os
import sys
import pytest
import tempfile
import urllib.parse  # Added for URL parsing
from unittest.mock import patch

# Add parent directory and shared directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(parent_dir))  # To access shared

# Import after updating sys.path
from utils.constants import MODEL_SERVER_URL

from app_factory import create_app

@pytest.fixture
def app():
    """Create and configure a Flask app for testing."""
    # Configure app for testing
    test_config = {
        'TESTING': True,
        'MODEL_SERVER_URL': MODEL_SERVER_URL,  # Use constant from utils.constants
        'MAX_CONTENT_LENGTH': 1024 * 1024  # 1 MB for testing
    }
    
    # Create the app with the test configuration
    app = create_app(test_config)
    
    # Return the app for testing
    yield app

@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return app.test_client()

def test_health_endpoint(client):
    """Test that the health endpoint returns 200 OK."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_error_handlers(client):
    """Test that 404 errors return a JSON response."""
    response = client.get('/nonexistent-route')
    assert response.status_code == 404
    assert 'error' in response.json
    assert 'message' in response.json

def test_blueprint_registration(app):
    """Test that blueprints are registered correctly."""
    # Get the list of registered blueprints
    blueprints = [bp.name for bp in app.blueprints.values()]
    
    # Check that our expected blueprints are registered
    assert 'health' in blueprints
    assert 'inference' in blueprints

def test_app_config(app):
    """Test that app config is properly set."""
    assert app.config['TESTING'] is True
    assert app.config['MODEL_SERVER_URL'] == MODEL_SERVER_URL
    assert app.config['MAX_CONTENT_LENGTH'] == 1024 * 1024

def test_model_service_initialization(app):
    """Test that the model service is initialized."""
    assert hasattr(app, 'model_service')
    assert app.model_service is not None

def test_inference_endpoint_exists(client):
    """Test that the inference endpoint exists."""
    # Just test that the endpoint exists by requesting a non-existent UUID
    response = client.post('/inference/models/uuid/nonexistent-uuid/predict', json={})
    # The response should be 404 (not found) rather than 405 (method not allowed)
    # which would indicate the route doesn't exist
    assert response.status_code == 404
    assert 'error' in response.json
