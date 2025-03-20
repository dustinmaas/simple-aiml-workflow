# Model Server Tests

Tests for the model server component.

## Test Coverage

- Database operations (add/retrieve/delete models and metadata)
- Storage operations (store/retrieve model files)
- API endpoints
- Model versioning

## Running the Tests

### Using run_tests.sh (Recommended)

```bash
# From the tests directory
./run_tests.sh

# Or from the project root
./model-server/tests/run_tests.sh
```

This script:
1. Stops existing containers
2. Removes volumes for a clean environment
3. Starts fresh containers
4. Runs all tests
5. Cleans up afterward

### Direct Testing

```bash
# From the project root
docker exec simple-aiml-workflow-model-server-1 python -m pytest /app/tests -v
```

## Test Implementation

Tests use:
- Temporary files for model creation
- The container's actual model storage directory
- Cleanup processes to remove test models
- Isolated test cases that don't depend on other tests
