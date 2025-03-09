#!/usr/bin/env python3
"""
Tests for the application factory of the inference server.

This module tests that the Flask application is properly configured and routes are registered.
"""

import os
import sys
import pytest
import tempfile
from unittest.mock import patch

# Add parent directory to path to import the app module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app_factory import create_app
from utils.model_cache import ModelCache

@pytest.fixture
def app():
    """Create and configure a Flask app for testing."""
    # Configure app for testing
    test_config = {
        'TESTING': True,
        'MODEL_SERVER_URL': 'http://localhost:5001',  # Mock URL
        'MAX_CONTENT_LENGTH': 1024 * 1024  # 1 MB for testing
    }
    
    # We need to patch the ModelCache to prevent it from making real HTTP requests
    with patch.object(ModelCache, '__init__', return_value=None) as mock_init:
        app = create_app(test_config)
        
        # Mock the cache dictionary as empty for testing
        app.model_cache.cache = {}
        
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

def test_cache_endpoint(client):
    """Test that the cache endpoint returns information about the cache."""
    response = client.get('/cache')
    assert response.status_code == 200
    assert 'cached_models' in response.json
    assert response.json['cached_models'] == 0  # Empty cache

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
    assert 'prediction' in blueprints

def test_app_config(app):
    """Test that app config is properly set."""
    assert app.config['TESTING'] is True
    assert app.config['MODEL_SERVER_URL'] == 'http://localhost:5001'
    assert app.config['MAX_CONTENT_LENGTH'] == 1024 * 1024

def test_model_cache_initialization(app):
    """Test that the model cache is initialized."""
    assert hasattr(app, 'model_cache')
    assert app.model_cache is not None
    assert hasattr(app.model_cache, 'cache')
    assert isinstance(app.model_cache.cache, dict)

def test_predict_endpoint_validation(client):
    """Test that the predict endpoint validates request data."""
    # Test missing features
    response = client.post('/predict', json={})
    assert response.status_code == 400
    assert 'error' in response.json
    assert 'features' in response.json['error']
    
    # Test empty features list
    response = client.post('/predict', json={'features': []})
    assert response.status_code == 400
    assert 'error' in response.json
    assert 'empty' in response.json['error'].lower()
    
    # Test invalid features type
    response = client.post('/predict', json={'features': 'not_a_list'})
    assert response.status_code == 400
    assert 'error' in response.json
    assert 'list' in response.json['error'].lower()
