#!/bin/bash
# Run tests for the model server in the Docker container using the test-specific compose file

echo "===== Starting model server test run with clean slate ====="

# Stop existing test containers if any
echo "Stopping test containers if running..."
sudo docker compose -f ../../docker-compose.test.yml down

# Remove test volumes to ensure clean state
echo "Removing test volumes for a clean slate..."
sudo docker volume rm aiml-model-data-test aiml-model-db-test || true

# Start containers with fresh test volumes
echo "Starting test containers..."
sudo docker compose -f ../../docker-compose.test.yml up -d model-server

# Wait for the service to be ready
echo "Waiting for service to be ready..."
sleep 5

# Run tests
echo "Running tests..."
sudo docker compose -f ../../docker-compose.test.yml exec model-server pytest -xvs /app/tests/

# Stop test containers
echo "Stopping test containers..."
sudo docker compose -f ../../docker-compose.test.yml down

echo "Test run complete!"
