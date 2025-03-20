# Inference Server

A server that provides an API for making predictions using ONNX models retrieved from the Model Server.

## Features

- Retrieve models from the Model Server using UUIDs or name/version combinations
- Cache models locally for improved performance
- Run inference on ONNX models using ONNXRuntime
- Automatically detect model input shapes and adapt input formats
- Support for model versioning

## Architecture

The Inference Server is built using Flask with these key components:

- **Application Factory**: Creates the Flask application (`app_factory.py`)
- **Routes**: Implements API endpoints (`routes/inference_routes.py`)
- **Model Service**: Handles model retrieval and caching (`utils/model_service.py`)
- **Model Cache**: Manages the local model cache (`utils/model_cache.py`)

## API Endpoints

See the [OpenAPI specification](./openapi.yaml) for detailed API documentation.

Key endpoints include:
- `POST /inference/models/{name}/latest/predict` - Predict using the latest model version
- `POST /inference/models/{name}/versions/{version}/predict` - Predict using a specific model version
- `POST /inference/models/uuid/{uuid}/predict` - Predict using a model by UUID

## Input Format

The inference server accepts input data in JSON format:

```json
{
  "input": [[10.0], [100.0]]
}
```

For single-feature models, use column vector format as shown above.

## Running Tests

### Using run_tests.sh (Recommended)

```bash
cd inference-server/tests
./run_tests.sh
```

This script manages the complete test workflow with a clean environment.

### Direct Testing

```bash
# Run all tests
sudo docker compose exec inference-server pytest /app/tests -v
```

## Environment Variables

- `MODEL_SERVER_URL`: URL of the Model Server (default: "http://model-server:5000")
- `MODEL_CACHE_DIR`: Cache directory for models (default: "/app/cache/models")
- `REQUEST_TIMEOUT`: Timeout for requests in seconds (default: 30)
- `MAX_CACHE_SIZE`: Maximum cache size (default: 10)

## Troubleshooting

### Model Input Format Issues
- Check that input format matches what the model expects
- Use column vector format for single-feature models
- Ensure input tensor name is correct (typically "input")

### Model Retrieval Issues
- Verify the Model Server is running and accessible
- Check if UUID extraction from file paths is working correctly
- Try using `run_tests.sh` to reset and recreate test models
