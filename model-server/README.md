# Model Server

A server that stores and serves ONNX models via a REST API, supporting semantic versioning and metadata management.

## Features

- Store and retrieve ONNX models using a REST API
- Support for semantic versioning (major.minor.patch)
- Separate metadata storage and retrieval
- UUID-based and name/version-based model access
- Simple ONNX model validation

## Architecture

The Model Server is built using Flask with these key components:

- **Application Factory**: Creates the Flask application (`app_factory.py`)
- **Database**: Stores model information and metadata (`utils/database.py`)
- **Storage**: Handles model file operations (`utils/storage.py`)
- **Routes**: Implements API endpoints (`routes/model_routes.py`)
- **Model Validation**: Validates ONNX models (`utils/model_validator.py`)

## API Endpoints

See the [OpenAPI specification](./openapi.yaml) for detailed API documentation.

Key endpoints include:
- `POST /models/{name}` - Upload a model
- `GET /models/{name}/versions/{version}` - Get a specific model version
- `GET /models/{name}/latest` - Get the latest model version
- `GET /models/uuid/{uuid}` - Get a model by UUID
- `GET /models/{name}/metadata` - Get model metadata

## Running Tests

### Using run_tests.sh (Recommended)

```bash
cd model-server/tests
./run_tests.sh
```

This script manages the complete test workflow with a clean environment.

### Direct Testing

```bash
# Run all tests
sudo docker compose exec model-server pytest /app/tests -v
```

## Environment Variables

- `MODEL_STORAGE_DIR`: Directory for storing model files (default: `/data/models`)
- `MODEL_DB_PATH`: Path to the SQLite database file (default: `/data/db/models.db`)
