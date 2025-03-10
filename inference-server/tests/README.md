# Inference Server Tests

This directory contains integration tests for the inference server. The tests are designed to run in the Docker container and use real ONNX inference with models from the model server.

## Enhanced Testing Approach

Unlike unit tests that use mocks and fixtures, these tests:

1. Connect to the real model server
2. Download real ONNX models
3. Run actual inference using ONNXRuntime
4. Test the UUID extraction pattern with real models
5. Dynamically detect model input shapes and adapt input formats
6. Verify consistent behavior with the same model architecture used in playground.py

This comprehensive testing approach ensures:
- The inference server can handle potential UUID mismatches between database records and storage files
- Input data is correctly formatted based on the model's expected dimensions
- Single-feature models receive properly formatted column vector inputs
- The same model architecture works consistently across all components

## Running Tests

Tests can be run using the improved `run_tests.sh` script in the parent directory. This script provides a comprehensive testing workflow:

```bash
./run_tests.sh
```

The script:

1. Stops existing containers
2. Removes model volumes for a clean slate
3. Starts containers with fresh volumes
4. Creates test models using the LinearRegressionModel from playground.py
5. Runs all tests with detailed output
6. Cleans up test models when complete

This approach ensures tests run in a clean environment with consistent test models, minimizing issues caused by leftover data from previous test runs.

## Test Files

- `test_model_service.py`: Tests the model service with the UUID extraction pattern and model caching
- `test_inference_routes.py`: Tests the API endpoints with real model inference
- `test_model_versioning.py`: Tests model versioning functionality with dynamic input handling

## Model Input Shape Detection

The tests now include code to detect and adapt to the ONNX model's input shape:

1. `get_model_input_shape`: Analyzes the ONNX model structure to extract dimension information
2. `create_input_data_for_shape`: Creates appropriately formatted input data based on the model's expected dimensions
3. Proper handling of single-feature models using column vector format (shape: [batch_size, 1])
4. Fallback strategies for when shape information is unavailable

Example of dynamic input handling:
```python
# Extract model shape
input_shape = get_model_input_shape(model_url)

# Create appropriate test data based on the model's shape
input_data = create_input_data_for_shape(input_shape)

# Make prediction request with the properly formatted data
response = requests.post(predict_url, json=input_data, headers={"Content-Type": "application/json"})
```

## Model Consistency

The tests use the LinearRegressionModel from playground.py to ensure consistency across all components:

```python
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)  # two input features, one output feature
        self.batch_norm = torch.nn.BatchNorm1d(2)  # Batch normalization for input features
        self.register_buffer('y_mean', torch.zeros(1))  # For output normalization
        self.register_buffer('y_std', torch.ones(1))    # For output normalization
```

This ensures that all tests work with the exact same model architecture that's used in the main application, providing more reliable and consistent test results.

## Test Dependencies

The tests require the following dependencies, which are installed in the Docker container:

- pytest
- onnxruntime
- numpy
- Flask
- requests
