# Inference Server

The Inference Server provides an API for making predictions using ONNX models retrieved from the Model Server. This version has been updated to align with the new model server architecture, implementing the UUID extraction pattern to handle potential UUID mismatches between database records and storage files. It now also supports dynamic model input shape detection to adapt to different model structures.

## Features

- Retrieve models from the Model Server using UUIDs, name/version, or latest version
- Handle UUID mismatches between database and storage using the extraction pattern
- Cache models locally to improve performance and reduce load on the Model Server
- Run inference on ONNX models using ONNXRuntime
- Return predictions with metadata
- Automatically detect model input shapes and adapt input formats accordingly
- Support column vector input format for single-feature models

## API Documentation

A complete OpenAPI specification is available in the [openapi.yaml](./openapi.yaml) file, which describes all available endpoints, request/response formats, and examples.

## UUID Extraction Pattern

The Inference Server implements the UUID extraction pattern to handle potential mismatches between database UUIDs and storage UUIDs:

1. When retrieving a model by UUID, the service:
   - Uses the provided UUID (database UUID) to request the model from the Model Server
   - Gets model detail information to obtain the file path, which contains the storage UUID
   - Extracts the storage UUID from the file path (e.g., from "/data/models/[storage_uuid].onnx")
   - Caches and uses the model file with the extracted storage UUID
   - Falls back to the database UUID if extraction fails

2. This pattern ensures that models can be correctly located and used even when there's a mismatch between the UUID used in the database and the UUID used in the storage filename.

## Model Input Handling

The Inference Server now implements dynamic model input shape detection:

1. When a model is loaded, the ONNX structure is analyzed to determine expected input dimensions
2. Input data is formatted to match the model's requirements
3. For single-feature models expecting column vectors, the input format is adjusted accordingly
4. Fallback strategies are in place when shape information is unavailable

Example input format for a single-feature model:
```json
{
  "input": [[10.0], [100.0]]
}
```

## Testing

### Automated Testing Workflow

The inference-server includes a comprehensive testing framework with a clean environment approach:

```bash
# Run tests with clean environment
./run_tests.sh
```

The `run_tests.sh` script:
1. Stops existing containers
2. Removes model volumes for a clean slate
3. Starts containers with fresh volumes
4. Creates test models using the LinearRegressionModel from playground.py
5. Runs all tests with detailed output
6. Cleans up test models when complete

### Model Consistency

The testing framework uses the same LinearRegressionModel from playground.py, ensuring consistency between development and testing. This model includes:

- Batch normalization on input features
- Linear regression layer with appropriate dimensions
- Mean and standard deviation normalization on outputs

This approach ensures that all components work with the same model architecture, providing more reliable test results.

## Troubleshooting

### Model Input Format Issues

If you encounter errors about input dimensions:
1. Check that your input format matches what the model expects (`{"input": [[10.0], [100.0]]}` for single-feature models)
2. Ensure you're using the correct input tensor name (typically "input")
3. For single-feature models, use column vector format where each value is in its own list

### Model Retrieval Issues

If models can't be retrieved from the Model Server:
1. Verify that the Model Server is running and accessible
2. Check if the UUID extraction from file paths is working correctly
3. Try running `run_tests.sh` to reset and recreate test models
4. Inspect the model cache directory for any corrupt models

## Configuration

The Inference Server uses environment variables for configuration:

- `MODEL_SERVER_URL` - URL of the Model Server (default: "http://model-server:5000")
- `MODEL_CACHE_DIR` - Directory to cache downloaded models (default: "/app/cache/models")
- `REQUEST_TIMEOUT` - Timeout for Model Server requests in seconds (default: 30)
- `MAX_CACHE_SIZE` - Maximum number of models to keep in cache (default: 10)
- `HOST` - Host to run the server on (default: "0.0.0.0")
- `PORT` - Port to run the server on (default: 5000)
- `DEBUG` - Enable debug mode (default: "False")
