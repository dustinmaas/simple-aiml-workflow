# Model Server Testing Guide

This guide explains how to run tests for the model server.

## Understanding Test Environment

The model server tests cover various aspects of the system:

1. **Database operations** - Tests for adding, retrieving, and deleting models and metadata
2. **Storage operations** - Tests for storing and retrieving model files
3. **API endpoints** - Tests for the REST API functionality
4. **Model versioning** - Tests for semantic versioning and version management

## Running Tests

All tests can be run directly against the live Model Server container:

```bash
# From the project root
docker exec simple-aiml-workflow-model-server-1 python -m pytest /app/tests -v
```

## Test Implementation

- Tests use temporary files for model creation and then upload them to the model server
- Tests are designed to clean up after themselves by deleting any models created during testing
- The container's actual model storage directory is used, with models being clearly named for testing and then deleted
- Each test is isolated and doesn't depend on the state created by other tests
