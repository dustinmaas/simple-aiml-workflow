#!/bin/bash
# Run tests for both model-server and inference-server by calling their respective test scripts

echo "===== Starting full test suite ====="

# Stop existing test containers if any
echo "Stopping test containers if running..."
sudo docker compose -f docker-compose.test.yml down

# Remove test volumes to ensure clean state
echo "Removing test volumes for a clean slate..."
sudo docker volume rm aiml-model-data-test aiml-model-db-test aiml-inference-cache-test || true

# Start containers with fresh test volumes
echo "Starting test containers..."
sudo docker compose -f docker-compose.test.yml up -d model-server inference-server

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 5

# Create test models for inference server tests
echo "Creating test models..."
sudo docker compose -f docker-compose.test.yml exec model-server python -c "import sys; sys.path.insert(0, '/app'); from shared.create_test_models import main; main()"

# Run model-server tests
echo "===== Running model-server tests ====="
sudo docker compose -f docker-compose.test.yml exec model-server pytest -xvs /app/tests/

# Run inference-server tests
echo "===== Running inference-server tests ====="
sudo docker compose -f docker-compose.test.yml exec inference-server pytest -xvs /app/tests/

# Clean up models via API
echo "Cleaning up test models..."
sudo docker compose -f docker-compose.test.yml exec model-server curl -s -X DELETE "http://localhost:5000/models/test_inference_model"
sudo docker compose -f docker-compose.test.yml exec model-server curl -s -X DELETE "http://localhost:5000/models/test_versioning_model"

# Stop test containers
echo "Stopping test containers..."
sudo docker compose -f docker-compose.test.yml down

echo "All tests complete!"
echo ""
echo "Note: You can run individual service tests with:"
echo "  model-server/tests/run_tests.sh    - For model server tests only"
echo "  inference-server/tests/run_tests.sh - For inference server tests only"
