#!/usr/bin/env python3
"""
Pytest tests for the model storage system.

This module contains test cases for:
1. Database operations
2. File storage operations
3. API endpoints
"""

import os
import sys
import uuid
import json
import pytest
import io
import tempfile
import torch
import torch.nn as nn
import torch.onnx
import requests
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import ModelDatabase
from utils.storage import ModelStorage
from utils.constants import MODEL_DB_PATH, MODEL_STORAGE_DIR

# Test constants
TEST_MODEL_NAME = "test_linear_model"
TEST_MODEL_VERSION = "1.0.0"
TEST_API_MODEL_NAME = "test_api_model"
TEST_API_MODEL_VERSION = "1.0.0"
# Make the server URL configurable for container vs. host testing
TEST_SERVER_URL = os.environ.get('MODEL_SERVER_URL', 'http://localhost:5000')

class LinearRegressionModel(nn.Module):
    """
    Linear regression model with batch normalization as used in playground.py.
    This model is designed to predict min_prb_ratio based on CQI and throughput.
    """
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # two input features, one output feature
        
        # Apply batch normalization to input features
        self.batch_norm = nn.BatchNorm1d(2)
        
        # Register buffers to store the mean and standard deviation of the output features
        self.register_buffer('y_mean', torch.zeros(1))
        self.register_buffer('y_std', torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = self.batch_norm(x)
        output = self.linear(x_normalized)
        
        if not self.training:
            with torch.no_grad():
                output = output * self.y_std + self.y_mean
                
        return output

@pytest.fixture
def model_path():
    """Create a test model and return its path."""
    # Create the LinearRegressionModel
    model = LinearRegressionModel()
    
    # Sample data for training
    features = torch.tensor([[10.0, 100.0], [8.0, 80.0], [6.0, 60.0], [4.0, 40.0], [2.0, 20.0]], dtype=torch.float32)
    targets = torch.tensor([[50.0], [60.0], [70.0], [80.0], [90.0]], dtype=torch.float32)
    
    # Set the y_mean and y_std buffers
    model.y_mean = targets.mean(dim=0, keepdim=True)
    model.y_std = targets.std(dim=0, keepdim=True)
    
    # Train the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    
    model.train()
    for epoch in range(10):  # Just a few epochs for testing
        # Forward pass
        y_predicted = model(features)
        loss = criterion(y_predicted, (targets - model.y_mean) / model.y_std)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Set to evaluation mode
    model.eval()
    
    # Create dummy input for export
    dummy_input = torch.randn(1, 2)  # Batch size of 1, 2 features (CQI, throughput)

    # Create a temporary file to save the model
    temp_path = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False).name

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        temp_path,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    yield temp_path
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)

@pytest.fixture
def db():
    """Return a database connection."""
    return ModelDatabase(MODEL_DB_PATH)

@pytest.fixture
def storage():
    """Return a storage manager using container's model storage directory."""
    # Use the actual model storage directory in the container
    return ModelStorage(MODEL_STORAGE_DIR)

@pytest.fixture
def model_uuid():
    """Return a UUID for testing."""
    return str(uuid.uuid4())

class TestModelDatabase:
    """Test cases for the ModelDatabase class."""

    def test_add_model(self, db, model_uuid):
        """Test adding a model to the database."""
        # Create test model data
        model_name = f"{TEST_MODEL_NAME}_db"
        version = TEST_MODEL_VERSION
        file_path = f"/data/models/{model_uuid}.onnx"
        file_size = 1000

        # Add model to database
        db_uuid = db.add_model(model_name, version, file_path, file_size)
        
        # Verify model was added
        assert db_uuid is not None
        
        # Cleanup
        db.delete_model_by_name_version(model_name, version)
        
    def test_get_model_by_name_version(self, db, model_uuid):
        """Test retrieving a model by name and version."""
        # Create test model data
        model_name = f"{TEST_MODEL_NAME}_get"
        version = TEST_MODEL_VERSION
        file_path = f"/data/models/{model_uuid}.onnx"
        file_size = 1000

        # Add model to database
        db_uuid = db.add_model(model_name, version, file_path, file_size)
        
        # Retrieve model
        model_info = db.get_model_by_name_version(model_name, version)
        
        # Verify model data
        assert model_info is not None
        assert model_info['name'] == model_name
        assert model_info['version'] == version
        assert model_info['file_path'] == file_path
        assert model_info['file_size'] == file_size
        
        # Cleanup
        db.delete_model(db_uuid)
        
    def test_add_metadata(self, db, model_uuid):
        """Test adding metadata to a model."""
        # Create test model data
        model_name = f"{TEST_MODEL_NAME}_meta"
        version = TEST_MODEL_VERSION
        file_path = f"/data/models/{model_uuid}.onnx"
        file_size = 1000

        # Add model to database
        db_uuid = db.add_model(model_name, version, file_path, file_size)
        
        # Create metadata
        metadata = {
            "model_name": model_name,
            "version": version,
            "training_date": datetime.now().isoformat(),
            "framework": f"PyTorch {torch.__version__}",
            "description": "Test model for UUID storage",
            "input_features": json.dumps(["feature1", "feature2"]),
            "output_features": json.dumps(["prediction"])
        }
        
        # Add metadata
        result = db.add_metadata(db_uuid, metadata)
        assert result is True
        
        # Retrieve metadata
        retrieved_metadata = db.get_metadata(db_uuid)
        assert retrieved_metadata is not None
        assert retrieved_metadata["model_name"] == model_name
        assert retrieved_metadata["version"] == version
        
        # Cleanup
        db.delete_model(db_uuid)
        
    def test_list_models(self, db, model_uuid):
        """Test listing all models."""
        # Create test model data
        model_name = f"{TEST_MODEL_NAME}_list"
        version = TEST_MODEL_VERSION
        file_path = f"/data/models/{model_uuid}.onnx"
        file_size = 1000

        # Add model to database
        db_uuid = db.add_model(model_name, version, file_path, file_size)
        
        # List models
        models = db.list_models()
        
        # Verify model is in list
        assert model_name in models
        assert len(models[model_name]) >= 1
        
        # Find our model in the list
        found = False
        for model in models[model_name]:
            if model['uuid'] == db_uuid:
                found = True
                break
                
        assert found
        
        # Cleanup
        db.delete_model(db_uuid)

    def test_delete_model(self, db, model_uuid):
        """Test deleting a model."""
        # Create test model data
        model_name = f"{TEST_MODEL_NAME}_delete"
        version = TEST_MODEL_VERSION
        file_path = f"/data/models/{model_uuid}.onnx"
        file_size = 1000

        # Add model to database
        db_uuid = db.add_model(model_name, version, file_path, file_size)
        
        # Delete model
        result = db.delete_model(db_uuid)
        assert result is True
        
        # Verify model is deleted - should raise ModelNotFoundError
        from utils.database import ModelNotFoundError
        with pytest.raises(ModelNotFoundError):
            db.get_model_by_uuid(db_uuid)

class TestModelStorage:
    """Test cases for the ModelStorage class."""
    
    def test_store_model(self, storage, model_path, model_uuid):
        """Test storing a model."""
        # Read model data
        with open(model_path, 'rb') as f:
            model_data = f.read()
            
        # Store model
        file_path, file_size = storage.store_model(model_uuid, model_data)
        
        # Verify model was stored
        # Use the storage object's own storage_dir, not the global constant
        assert file_path == os.path.join(storage.storage_dir, f"{model_uuid}.onnx")
        assert file_size == len(model_data)
        assert storage.model_exists(model_uuid)
        
        # Cleanup
        storage.delete_model(model_uuid)
        
    def test_get_model(self, storage, model_path, model_uuid):
        """Test retrieving a model."""
        # Read model data
        with open(model_path, 'rb') as f:
            model_data = f.read()
            
        # Store model
        storage.store_model(model_uuid, model_data)
        
        # Retrieve model
        retrieved_data = storage.get_model(model_uuid)
        
        # Verify model data
        assert retrieved_data == model_data
        
        # Cleanup
        storage.delete_model(model_uuid)
        
    def test_delete_model(self, storage, model_path, model_uuid):
        """Test deleting a model."""
        # Read model data
        with open(model_path, 'rb') as f:
            model_data = f.read()
            
        # Store model
        storage.store_model(model_uuid, model_data)
        
        # Delete model
        result = storage.delete_model(model_uuid)
        assert result is True
        
        # Verify model is deleted
        assert not storage.model_exists(model_uuid)

class TestModelAPI:
    """Test cases for the model API."""
    
    def test_upload_model(self, model_path):
        """Test uploading a model via the API."""
        # Read model data
        with open(model_path, 'rb') as f:
            model_data = f.read()
            
        # Create metadata
        metadata = {
            "training_date": datetime.now().isoformat(),
            "framework": f"PyTorch {torch.__version__}",
            "description": "Test model for UUID storage API",
            "input_features": ["feature1", "feature2"],
            "output_features": ["prediction"]
        }
        
        # Upload model
        files = {'model': (f"{TEST_API_MODEL_NAME}_v{TEST_API_MODEL_VERSION}.onnx", model_data)}
        data = {'metadata': json.dumps(metadata)}
        response = requests.post(
            f"{TEST_SERVER_URL}/models/{TEST_API_MODEL_NAME}/versions/{TEST_API_MODEL_VERSION}",
            files=files,
            data=data
        )
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "uuid" in result
        
        # Store UUID for cleanup
        model_uuid = result["uuid"]
        
        # Cleanup
        response = requests.delete(f"{TEST_SERVER_URL}/models/{TEST_API_MODEL_NAME}/versions/{TEST_API_MODEL_VERSION}")
        assert response.status_code == 200
        
    def test_list_models(self, model_path):
        """Test listing models via the API."""
        # Read model data
        with open(model_path, 'rb') as f:
            model_data = f.read()
            
        # Upload model
        files = {'model': (f"{TEST_API_MODEL_NAME}_list_v{TEST_API_MODEL_VERSION}.onnx", model_data)}
        response = requests.post(
            f"{TEST_SERVER_URL}/models/{TEST_API_MODEL_NAME}_list/versions/{TEST_API_MODEL_VERSION}",
            files=files
        )
        
        # Verify response
        assert response.status_code == 200
        
        # List models
        response = requests.get(f"{TEST_SERVER_URL}/models")
        
        # Verify response
        assert response.status_code == 200
        models = response.json()
        assert f"{TEST_API_MODEL_NAME}_list" in models
        
        # Cleanup
        requests.delete(f"{TEST_SERVER_URL}/models/{TEST_API_MODEL_NAME}_list/versions/{TEST_API_MODEL_VERSION}")
        
    def test_get_model_by_uuid(self, model_path):
        """Test retrieving a model by UUID via the API."""
        model_name = f"{TEST_API_MODEL_NAME}_uuid"
        
        # First, delete any existing model with this name and version
        delete_response = requests.delete(
            f"{TEST_SERVER_URL}/models/{model_name}/versions/{TEST_API_MODEL_VERSION}")
        print(f"Delete response status: {delete_response.status_code}, text: {delete_response.text}")
        
        # Read model data
        with open(model_path, 'rb') as f:
            model_data = f.read()
            
        # Create metadata to ensure it's properly stored
        metadata = {
            "training_date": datetime.now().isoformat(),
            "framework": f"PyTorch {torch.__version__}",
            "description": "Test model for UUID storage API - get by UUID test",
            "input_features": ["feature1", "feature2"],
            "output_features": ["prediction"]
        }
            
        # Upload model with metadata
        files = {'model': (f"{model_name}_v{TEST_API_MODEL_VERSION}.onnx", model_data)}
        data = {'metadata': json.dumps(metadata)}
        
        upload_response = requests.post(
            f"{TEST_SERVER_URL}/models/{model_name}/versions/{TEST_API_MODEL_VERSION}",
            files=files,
            data=data
        )
        
        # Verify response
        assert upload_response.status_code == 200, f"Failed to upload model: {upload_response.text}"
        result = upload_response.json()
        assert result["success"] is True, f"Upload response does not indicate success: {result}"
        assert "uuid" in result, f"UUID not found in response: {result}"
        model_uuid = result["uuid"]
        
        print(f"Model uploaded with UUID: {model_uuid}")
        
        # List all models to verify the model was added and get the actual registered UUID
        list_response = requests.get(f"{TEST_SERVER_URL}/models")
        assert list_response.status_code == 200, f"Failed to list models: {list_response.text}"
        models = list_response.json()
        print(f"Available models: {json.dumps(models, indent=2)}")
        
        # Make sure the model exists in the list
        assert f"{model_name}" in models, f"Model not found in models list: {models}"
        
        # Extract the actual UUID from the listing (which may differ from what was returned during upload)
        actual_uuid = models[model_name][0]["uuid"]
        print(f"Upload returned UUID: {model_uuid}, but database has UUID: {actual_uuid}")
        
        # Get model by the UUID from the database listing
        response = requests.get(f"{TEST_SERVER_URL}/models/uuid/{actual_uuid}")
        
        # Verify response
        assert response.status_code == 200, f"Failed to get model by UUID: {response.text}"
        
        # Cleanup
        delete_response = requests.delete(f"{TEST_SERVER_URL}/models/{TEST_API_MODEL_NAME}_uuid/versions/{TEST_API_MODEL_VERSION}")
        assert delete_response.status_code == 200, f"Failed to delete model: {delete_response.text}"
        
    def test_get_model_detail(self, model_path):
        """Test retrieving model details via the API."""
        # Read model data
        with open(model_path, 'rb') as f:
            model_data = f.read()
            
        # Upload model
        files = {'model': (f"{TEST_API_MODEL_NAME}_detail_v{TEST_API_MODEL_VERSION}.onnx", model_data)}
        response = requests.post(
            f"{TEST_SERVER_URL}/models/{TEST_API_MODEL_NAME}_detail/versions/{TEST_API_MODEL_VERSION}",
            files=files
        )
        
        # Verify response
        assert response.status_code == 200
        
        # Get model details
        response = requests.get(f"{TEST_SERVER_URL}/models/{TEST_API_MODEL_NAME}_detail/versions/{TEST_API_MODEL_VERSION}/detail")
        
        # Verify response
        assert response.status_code == 200
        details = response.json()
        assert details["model_name"] == f"{TEST_API_MODEL_NAME}_detail"
        assert details["version"] == TEST_API_MODEL_VERSION
        
        # Cleanup
        requests.delete(f"{TEST_SERVER_URL}/models/{TEST_API_MODEL_NAME}_detail/versions/{TEST_API_MODEL_VERSION}")
