# Model Server

## Overview

The Model Server is a core component of the PyTorch AI/ML Workflow. It stores and serves ONNX models via a REST API, supporting semantic versioning and metadata management. This component acts as a central repository for trained models, allowing other services to retrieve models for inference or analysis.

## Architecture

The Model Server is built using Flask and follows a modular design:

- **Application Factory**: Creates and configures the Flask application (`app_factory.py`)
- **Database**: SQLite-based storage for model information and metadata (`utils/database.py`)
- **Storage**: File-based storage for model binaries (`utils/storage.py`)
- **Routes**: API endpoints for model management (`routes/model_routes.py`)
- **Model Validation**: ONNX model validation utilities (`utils/model_validator.py`)
- **Constants**: Centralized configuration values (`utils/constants.py`)

## UUID-Based Storage System

The Model Server implements a robust UUID management pattern to handle potential mismatches between database UUIDs and storage UUIDs:

1. When a model is uploaded, a single UUID is generated for both database and storage
2. Models are stored in the filesystem with the filename being the UUID
3. The file path (containing the storage UUID) is stored in the database record
4. When retrieving models, the system extracts the storage UUID from the file path
5. This ensures reliable model retrieval even in edge cases where UUIDs might differ

## API Documentation

The Model Server provides a comprehensive REST API for model management. For detailed endpoint documentation, refer to:

- **[OpenAPI Specification](./openapi.yaml)**

The API supports:
- Model versioning with semantic versioning support
- Separate metadata management
- Both name/version-based and UUID-based access patterns
- Detailed model information retrieval

## Running Tests

There are multiple approaches to running tests for the Model Server:

### Method 1: Using the run_tests.sh Script (Recommended)

The inference-server component includes a `run_tests.sh` script that manages the complete test workflow with a clean environment:

```bash
cd /var/tmp/simple-aiml-workflow/inference-server
./run_tests.sh
```

This script:
1. Stops containers
2. Removes model volumes for a clean slate
3. Restarts services
4. Creates test models using the LinearRegressionModel
5. Runs tests
6. Cleans up test artifacts

### Method 2: Direct Container Testing

Run tests directly against the running Model Server container:

```bash
# Run all tests
sudo docker compose exec model-server pytest /app/tests -v
```

### Method 3: Using the Docker exec Command

If you prefer the docker exec approach:

```bash
docker exec simple-aiml-workflow-model-server-1 python -m pytest /app/tests -v
```

### Testing with LinearRegressionModel

The testing framework now uses the same LinearRegressionModel from playground.py, ensuring consistency across all environments. This approach:

1. Makes tests more reliable by using the same model architecture everywhere
2. Automatically detects and adapts to the model's input shape requirements
3. Properly handles single-feature models with column vector format inputs

## Environment Variables

- `MODEL_STORAGE_DIR`: Directory for storing model files (default: `/data/models`)
- `MODEL_DB_PATH`: Path to the SQLite database file (default: `/data/db/models.db`)

## Dependencies

- Flask: Web framework
- ONNX: Open Neural Network Exchange format
- ONNX Runtime: Runtime for ONNX models
- SQLite: Database for model metadata and references

## Troubleshooting

### "Model file not found" errors

If tests are failing with "Model file not found" errors, check:

1. Proper UUID extraction from file paths
2. Database/storage UUID mismatch handling
3. Volume mounting in Docker configuration
4. Try using the `run_tests.sh` script which starts with clean volumes

### Model Dimension Mismatch

If you see errors about input dimensions not matching, check:

1. The input format in your request matches what the model expects (typically column vector format for single-feature models)
2. The get_model_input_shape function is extracting the correct shape information
3. Your input data is properly formatted as `{"input": [[10.0], [100.0]]}` for single-feature models

### Database Method Missing

If you see errors like "'ModelDatabase' object has no attribute 'add_model_with_uuid'", ensure:

1. The ModelDatabase class in database.py includes all required methods
2. Methods are properly indented to be part of the class
3. Any changes to database.py are reflected in the running container (restart may be required)
