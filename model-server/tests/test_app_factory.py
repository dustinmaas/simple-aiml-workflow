#!/usr/bin/env python3
"""
Tests for the application factory of the model server.

This module tests that the Flask application is properly configured and routes are registered.
"""

import os
import sys
import pytest
import tempfile

# Add parent directory and shared directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(parent_dir))  # To access shared

from app_factory import create_app

@pytest.fixture
def app():
    """Create and configure a Flask app for testing."""
    # Create a temporary directory for test models
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configure app for testing
        test_config = {
            'TESTING': True,
            'MODELS_DIR': temp_dir,
            'MAX_CONTENT_LENGTH': 1024 * 1024  # 1 MB for testing
        }
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
    assert response.json['status'] == 'ok'

def test_status_endpoint(client):
    """Test that the status endpoint returns storage info."""
    response = client.get('/status')
    assert response.status_code == 200
    assert 'status' in response.json
    assert 'storage' in response.json

def test_models_endpoint(client):
    """Test that the models endpoint returns a dictionary."""
    response = client.get('/models')
    assert response.status_code == 200
    # Initially there are no models, so we just get an empty dict
    assert isinstance(response.json, dict)

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
    assert 'models' in blueprints

def test_app_config(app):
    """Test that app config is properly set."""
    assert app.config['TESTING'] is True
    assert 'MODELS_DIR' in app.config
    assert app.config['MAX_CONTENT_LENGTH'] == 1024 * 1024
