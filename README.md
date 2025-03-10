# PyTorch AI/ML Workflow

This project demonstrates a complete AI/ML workflow using PyTorch for training and inference in a microservices architecture.

## Architecture

The project consists of the following components:

1. **Model Server** - Stores and serves trained PyTorch models
2. **Inference Server** - Provides an API for making predictions using the models
3. **Analysis and Training Environment** - Jupyter notebook environment for data analysis and model training
4. **Experiment Runner** - Tool for running end-to-end tests of the workflow

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.8+

### Running the Services

To start all services:

```bash
docker-compose up --build
```

This will:
1. Build and start the model server on port 5001
2. Build and start the inference server on port 5002  
3. Build and start the Jupyter notebook server on port 8888

### Running Components Individually

You can also run specific components:

```bash
# Start only the model and inference servers
docker-compose up model-server inference-server

# Start the Jupyter notebook environment
docker-compose up analysis-and-training

```

## Using the Project

### Generating data with the experiment runner
You can run a test experiment with:

```bash
cd /var/tmp/simple-aiml-workflow/experiment-runner
./runner --config=test
```

### Training a Model

1. Access the Jupyter notebook at http://localhost:8888
2. Open `analysis-and-training/notebooks/playground.py` and use it as a starting point for data analysis and model creation/training/export.

### Making Predictions

Send a POST request to the inference server:

```bash
curl -X POST http://localhost:5002/inference/models/test_inference_model/latest/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [[10.0], [100.0]]}'
```

You can also specify a model version:

```bash
curl -X POST http://localhost:5002/inference/models/test_inference_model/versions/1.0.0/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [[10.0], [100.0]]}'
```

Or use a model UUID:

```bash
curl -X POST http://localhost:5002/inference/models/uuid/<model-uuid>/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [[10.0], [100.0]]}'
```

Note: The input format is a column vector where each row is a separate input sample. The model expects inputs in the format `[[value1], [value2], ...]`.

### Running Tests

There are several ways to run tests in this project:

#### Component Tests

For running tests on individual components, use the component-specific test scripts:

```bash
# Run inference-server tests with clean environment
cd inference-server
./run_tests.sh

# Run model-server tests directly with pytest
cd model-server
sudo docker compose exec model-server pytest -xvs
```

The `run_tests.sh` script provides a comprehensive testing workflow that:
1. Stops existing containers
2. Removes model volumes for a clean slate
3. Starts containers with fresh volumes
4. Creates test models using the LinearRegressionModel from playground.py
5. Runs all tests with detailed output
6. Cleans up test models when complete

#### End-to-End Tests

You can run the end-to-end test script for workflow validation:

```bash
docker-compose run experiment-runner python /app/test_torch_inference.py
```

Or modify the Docker Compose configuration to run in test mode automatically:

```yaml
command: python /app/run_experiment.py --mode=test
```

#### Test Model Consistency

The testing framework now uses the same LinearRegressionModel from playground.py across all test environments, ensuring consistency between development and testing. This model includes:

- Batch normalization on input features
- Linear regression layer with appropriate dimensions
- Mean and standard deviation normalization on outputs
- Dynamic input shape detection to adapt to different model configurations

This approach ensures that all components work with the same model architecture, providing more reliable test results.
