#!/bin/bash
# Run tests for the inference server in the Docker container using the test-specific compose file

echo "===== Starting test run with clean slate ====="

# Stop existing test containers if any
echo "Stopping test containers if running..."
sudo docker compose -f ../../docker-compose.test.yml down

# Remove test volumes to ensure clean state
echo "Removing test volumes for a clean slate..."
sudo docker volume rm aiml-model-data-test aiml-model-db-test aiml-inference-cache-test || true

# Start containers with fresh test volumes
echo "Starting test containers..."
sudo docker compose -f ../../docker-compose.test.yml up -d model-server inference-server

# Wait for them to be ready
echo "Waiting for services to be ready..."
sleep 5

# Create test models
echo "Creating test models..."
sudo docker compose -f ../../docker-compose.test.yml exec model-server python -c "import sys; sys.path.insert(0, '/app'); from shared.create_test_models import main; main()"

# Run tests
echo "Running tests..."
sudo docker compose -f ../../docker-compose.test.yml exec inference-server pytest -xvs /app/tests/

# Clean up models via API
echo "Cleaning up test models..."
sudo docker compose -f ../../docker-compose.test.yml exec model-server curl -s -X DELETE "http://localhost:5000/models/test_inference_model"
sudo docker compose -f ../../docker-compose.test.yml exec model-server curl -s -X DELETE "http://localhost:5000/models/test_versioning_model"

# Stop test containers
echo "Stopping test containers..."
sudo docker compose -f ../../docker-compose.test.yml down

echo "Test run complete!"
