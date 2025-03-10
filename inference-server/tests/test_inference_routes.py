import os
import sys
import json
import pytest
import requests
from flask import url_for

# Add parent directory and shared directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(parent_dir))  # To access shared

from app_factory import create_app
from utils.constants import MODEL_SERVER_URL
from shared.test_utils import (
    download_and_analyze_model,
    create_input_data_for_shape,
    check_model_server_availability,
    get_available_models,
    get_model_detail,
    extract_storage_uuid
)

class TestInferenceAPI:
    def setup_method(self):
        app = create_app({"TESTING": True})
        self.client = app.test_client()
        
        # Check model server availability
        if not check_model_server_availability(MODEL_SERVER_URL):
            pytest.skip("Model server is not available")
        
        # Get available models
        models = get_available_models(MODEL_SERVER_URL)
        if not models:
            pytest.skip("No models available on model server")
        
        model_name = list(models.keys())[0]
        model_info = models[model_name][0]
        
        self.test_model_name = model_name
        self.test_model_version = model_info['version']
        self.test_model_uuid = model_info['uuid']
        
        # Get model details
        model_detail = get_model_detail(MODEL_SERVER_URL, self.test_model_uuid)
        if not model_detail:
            pytest.skip(f"Cannot get model detail for UUID {self.test_model_uuid}")
        
        file_path = model_detail.get('path', '')
        if not file_path:
            pytest.skip("Model file path not found in detail response")
        
        self.storage_uuid = extract_storage_uuid(file_path)
    
    def test_health_endpoint(self):
        response = self.client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['server'] == 'inference'
        assert 'timestamp' in data
    
    def test_inference_with_uuid(self):
        # Get the model and analyze its input shape
        input_shape, _ = download_and_analyze_model(
            f"{MODEL_SERVER_URL}/models/uuid/{self.test_model_uuid}"
        )
        
        # Create test data based on the model shape
        input_data = create_input_data_for_shape(input_shape)
                
        # Make prediction request
        response = self.client.post(
            f'/inference/models/uuid/{self.test_model_uuid}/predict',
            json=input_data,
            content_type='application/json'
        )
        
        # Verify response
        
        assert response.status_code == 200, f"Prediction failed: {response.data.decode('utf-8')}"
        data = json.loads(response.data)
        
        # Check response structure
        assert 'model_uuid' in data
        assert 'prediction' in data
        assert 'processing_time_seconds' in data
        
        # Verify correct model was used
        assert data['model_uuid'] == self.test_model_uuid
        
        # Verify prediction has expected structure
        assert isinstance(data['prediction'], dict)
        assert len(data['prediction']) > 0
    
    
    def test_inference_by_name_version(self):
        # Get the model and analyze its input shape
        input_shape, _ = download_and_analyze_model(
            f"{MODEL_SERVER_URL}/models/{self.test_model_name}/versions/{self.test_model_version}"
        )
        
        # Create test data based on the model shape
        input_data = create_input_data_for_shape(input_shape)
        
        # Make prediction request
        response = self.client.post(
            f'/inference/models/{self.test_model_name}/versions/{self.test_model_version}/predict',
            json=input_data,
            content_type='application/json'
        )
        
        # Verify response
        assert response.status_code == 200, f"Prediction failed: {response.data.decode('utf-8')}"
        data = json.loads(response.data)
        
        # Check response structure
        assert 'model_name' in data
        assert 'model_version' in data
        assert 'prediction' in data
        assert 'processing_time_seconds' in data
        
        # Verify correct model was used
        assert data['model_name'] == self.test_model_name
        assert data['model_version'] == self.test_model_version
        
        # Verify prediction has expected structure
        assert isinstance(data['prediction'], dict)
        assert len(data['prediction']) > 0
    
    def test_model_list(self):
        response = self.client.get('/inference/models/list')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert len(data) > 0
        
        assert self.test_model_name in data
    
    def test_cache_clear(self):
        response = self.client.post('/inference/cache/clear')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'message' in data
        assert 'Model cache cleared successfully' in data['message']
    
    def test_error_cases(self):
        # Test invalid request methods
        response = self.client.get(f'/inference/models/uuid/{self.test_model_uuid}/predict')
        assert response.status_code == 405  # Method Not Allowed
