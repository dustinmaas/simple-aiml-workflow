#!/bin/bash
# Run tests for the inference server in the Docker container

echo "===== Starting test run with clean slate ====="

# Stop existing containers
echo "Stopping containers..."
sudo docker compose down

# Remove model volumes
echo "Removing volumes for a clean slate..."
sudo docker volume rm aiml-model-data aiml-model-db || true

# Start containers with fresh volumes
echo "Starting containers..."
sudo docker compose up -d model-server inference-server

# Wait for them to be ready
echo "Waiting for services to be ready..."
sleep 5

# Create test models
echo "Creating test models..."
sudo docker compose exec model-server python create_test_models.py

# Run tests
echo "Running tests..."
sudo docker compose exec inference-server pytest -xvs /app/tests/

# Clean up models via API
echo "Cleaning up test models..."
sudo docker compose exec model-server curl -s -X DELETE "http://localhost:5000/models/test_inference_model"
sudo docker compose exec model-server curl -s -X DELETE "http://localhost:5000/models/test_versioning_model"

echo "Test run complete!"
