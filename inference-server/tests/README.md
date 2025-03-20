# Inference Server Tests

Tests for the inference server component.

## Test Coverage

- Model retrieval from Model Server
- UUID extraction pattern
- Dynamic model input shape detection
- API endpoints for predictions
- Model versioning support
- Model caching

## Running the Tests

### Using run_tests.sh (Recommended)

```bash
# From the tests directory
./run_tests.sh

# Or from the project root
./inference-server/tests/run_tests.sh
```

This script:
1. Stops existing containers
2. Removes volumes for a clean environment
3. Starts fresh containers
4. Creates test models
5. Runs all tests
6. Cleans up afterward

### Direct Testing

```bash
# From the project root
docker exec simple-aiml-workflow-inference-server-1 python -m pytest /app/tests -v
```

## Test Implementation

Tests use:
- Real model retrieval from the Model Server
- ONNX Runtime for actual inference
- Dynamic shape detection to format input data
- The same LinearRegressionModel as in playground.py
- Cleanup processes to remove test models
