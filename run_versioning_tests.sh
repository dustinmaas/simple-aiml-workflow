#!/bin/bash
# Script to run the model versioning tests in Docker containers

# Default is to rebuild and redeploy
REBUILD=true
RESTART=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-rebuild)
      REBUILD=false
      shift
      ;;
    --no-restart)
      RESTART=false
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --no-rebuild    Skip rebuilding containers"
      echo "  --no-restart    Skip restarting containers"
      echo "  --help, -h      Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run '$0 --help' for usage information"
      exit 1
      ;;
  esac
done

# Check if we need to stop and restart containers
if $RESTART; then
  echo "Stopping existing containers..."
  sudo docker compose down
  
  # Rebuild the containers if needed
  if $REBUILD; then
    echo "Rebuilding containers..."
    sudo docker compose build model-server inference-server
  fi
  
  # Start the containers in detached mode
  echo "Starting services..."
  sudo docker compose up -d model-server inference-server
  
  # Wait for services to initialize
  echo "Waiting for services to initialize..."
  sleep 5
else
  echo "Skipping container restart as requested..."
  # Check if services are running
  RUNNING_SERVICES=$(sudo docker compose ps --services --filter "status=running")
  if [[ ! "$RUNNING_SERVICES" =~ "model-server" ]] || [[ ! "$RUNNING_SERVICES" =~ "inference-server" ]]; then
    echo "Error: Some required services are not running. Please restart services or remove --no-restart flag."
    exit 1
  fi
fi

# Clean up any existing test models more reliably via the container
echo "Cleaning up any existing test models..."
sudo docker compose exec model-server bash -c '
  # Check if models directory exists and if test_model files are present
  echo "Looking for existing test models in container..."
  if ls /app/models/test_model_v*.onnx 2>/dev/null; then
    echo "Found test model files, removing..."
    rm -f /app/models/test_model_v*.onnx
    echo "Test model files removed."
  else
    echo "No test model files found in container."
  fi
'
# Wait a moment for cleanup to complete
sleep 2

# Run model-server tests
echo -e "\n============= Testing Model Server Versioning =============\n"
sudo docker compose exec model-server python /app/tests/test_model_versioning.py

# If model-server tests succeed, run inference-server tests
if [ $? -eq 0 ]; then
    echo -e "\n============= Testing Inference Server Versioning =============\n"
    sudo docker compose exec inference-server python /app/tests/test_inference_versioning.py
else
    echo "Model server tests failed. Skipping inference server tests."
fi

echo -e "\n============= Test Results =============\n"
if [ $? -eq 0 ]; then
    echo "All versioning tests completed successfully!"
else
    echo "Some tests failed. See above for details."
fi
