import os
import sys
import json
import pytest
import requests
import time
import onnxruntime as ort
import numpy as np
from flask import url_for
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_factory import create_app
from utils.constants import MODEL_SERVER_URL

class TestInferenceAPI:
    def setup_method(self):
        app = create_app({"TESTING": True})
        self.client = app.test_client()
        
        try:
            response = requests.get(f"{MODEL_SERVER_URL}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Model server is not available")
        except (requests.ConnectionError, requests.Timeout):
            pytest.skip("Model server is not available")
        
        response = requests.get(f"{MODEL_SERVER_URL}/models")
        if response.status_code != 200 or not response.json():
            pytest.skip("No models available on model server")
        
        models = response.json()
        model_name = list(models.keys())[0]
        model_info = models[model_name][0]
        
        self.test_model_name = model_name
        self.test_model_version = model_info['version']
        self.test_model_uuid = model_info['uuid']
        
        detail_response = requests.get(f"{MODEL_SERVER_URL}/models/uuid/{self.test_model_uuid}/detail")
        if detail_response.status_code != 200:
            pytest.skip(f"Cannot get model detail: {detail_response.text}")
        
        model_detail = detail_response.json()
        file_path = model_detail.get('path', '')
        
        if not file_path:
            pytest.skip("Model file path not found in detail response")
        
        self.storage_uuid = os.path.basename(file_path).replace(".onnx", "")
    
    def test_health_endpoint(self):
        response = self.client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['server'] == 'inference'
        assert 'timestamp' in data
    
    def test_inference_with_uuid(self):
        # Get the model from the server and inspect its structure
        import onnx
        model_metadata = None
        input_shape = None
        
        # Get detailed metadata from the model server
        detail_response = requests.get(f"{MODEL_SERVER_URL}/models/uuid/{self.test_model_uuid}/detail")
        if detail_response.status_code == 200:
            model_detail = detail_response.json()
            metadata_response = requests.get(f"{MODEL_SERVER_URL}/models/uuid/{self.test_model_uuid}/metadata")
            if metadata_response.status_code == 200:
                model_metadata = metadata_response.json()
        
        # Get the model itself to inspect shape
        try:
            model_response = requests.get(f"{MODEL_SERVER_URL}/models/uuid/{self.test_model_uuid}")
            if model_response.status_code == 200:
                # Save the model to a temporary file
                temp_model_path = "/tmp/temp_model.onnx"
                with open(temp_model_path, "wb") as f:
                    f.write(model_response.content)
                
                # Load the model and analyze its input shape
                onnx_model = onnx.load(temp_model_path)
                input_tensor = onnx_model.graph.input[0]
                input_shape = []
                for dim in input_tensor.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        input_shape.append(dim.dim_value)
                    else:
                        input_shape.append(None)  # Dynamic dimension
                
                print(f"Model input shape from ONNX: {input_shape}")
                
                # Create properly formatted test data based on the actual model shape
                input_data = self._create_input_data_for_shape(input_shape)
                
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
                return
            
        except Exception as e:
            print(f"Error analyzing model: {e}")
        
        # Fallback if we couldn't determine the shape from ONNX
        # Use our best guess for a Linear Regression model with 2 features
        # This is a fallback plan in case the above method fails
        input_data = {"input": [[10.0, 100.0]]}  # batched format with 2 features
        
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
    
    def _create_input_data_for_shape(self, shape):
        """Create properly formatted input data based on the model's expected shape."""
        # Default test values
        test_values = [10.0, 100.0]
        
        if not shape or len(shape) < 2:
            # Fallback: Use batched format with 2 features
            return {"input": [[10.0, 100.0]]}
            
        if shape[0] is None:  # First dimension is batch size (typically None/dynamic)
            if shape[1] == 1:
                # Model expects a column vector [batch_size, 1]
                return {"input": [[v] for v in test_values]}
            elif shape[1] == 2:
                # Model expects [batch_size, 2]
                return {"input": [test_values]}
            else:
                # Unknown feature count, use our test values
                return {"input": [test_values]}
        else:
            # Fixed batch size
            if shape[1] == 1:
                # Model expects a column vector with fixed batch size
                return {"input": [[v] for v in test_values[:shape[0]]]}
            else:
                # Model expects a fixed batch size with multiple features
                return {"input": [test_values[:shape[1]]]}
    
    def test_inference_by_name_version(self):
        # Get the model from the server and inspect its structure
        import onnx
        model_metadata = None
        input_shape = None
        
        # Get the model itself to inspect shape
        try:
            model_response = requests.get(f"{MODEL_SERVER_URL}/models/{self.test_model_name}/versions/{self.test_model_version}")
            if model_response.status_code == 200:
                # Save the model to a temporary file
                temp_model_path = "/tmp/temp_model_version.onnx"
                with open(temp_model_path, "wb") as f:
                    f.write(model_response.content)
                
                # Load the model and analyze its input shape
                onnx_model = onnx.load(temp_model_path)
                input_tensor = onnx_model.graph.input[0]
                input_shape = []
                for dim in input_tensor.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        input_shape.append(dim.dim_value)
                    else:
                        input_shape.append(None)  # Dynamic dimension
                
                print(f"Model input shape from ONNX: {input_shape}")
                
                # Create properly formatted test data based on the actual model shape
                input_data = self._create_input_data_for_shape(input_shape)
                
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
                return
                
        except Exception as e:
            print(f"Error analyzing model: {e}")
            
        # Fallback if we couldn't determine the shape from ONNX
        # Use our best guess for a Linear Regression model with 2 features
        input_data = {"input": [[10.0, 100.0]]}  # batched format with 2 features
        
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
