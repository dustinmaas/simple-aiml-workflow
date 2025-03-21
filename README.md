# POWDER Simple AI/ML Workflow with PyTorch

A complete AI/ML workflow using PyTorch for training and inference in a microservices architecture orchestrated with Docker Compose. This project provides a modular and containerized solution for data generation, model training, model storage, and inference.

## Architecture

The project consists of the following components:

1. **Experiment Runner** - Orchestrates end-to-end tests and experiments
2. **Model Server** - Stores and serves trained PyTorch models exported to ONNX format
3. **Inference Server** - Provides an API for making predictions using the models
4. **Analysis and Training** - Jupyter notebook environment for data analysis and model development
5. **Data Lake (InfluxDB)** - Time series database for storing metrics and other experiment data
6. **Shared Utilities** - Common ML functions and models shared between components

## Key Features

- Complete, reproducible AI/ML workflow from data generation to inference
- Model versioning with metadata storage
- ONNX format for model interoperability
- Containerized deployment with Docker Compose
- Hybrid storage strategy supporting both local (Model Server) and cloud-based (Hugging Face) options

## Getting Started

### Running the Services

To start all services:

```bash
sudo docker compose up -d
```

This will:
1. Start the model server (port 80; docker host port 5001)
2. Start the inference server (port 80; docker host port 5002)
3. Start the Jupyter notebook server (port 8888; docker host port 8888)
4. Start data lake server for storing metrics and experiment data

**Note:** All Docker commands must be run with `sudo` due to permission requirements atm.

### Running Components Individually

You can also run specific components:

```bash
# Start only the model and inference servers
sudo docker compose up -d model-server inference-server

# Start the Jupyter notebook environment
sudo docker compose up -d analysis-and-training
```

## Using the Project

### Data Generation and Analysis

1. Use the Experiment Runner to generate datasets
2. Access the Jupyter notebook at http://localhost:8888
3. Use the notebooks in `analysis-and-training/notebooks/` for data analysis and model development

### Model Development and Training

See the [Analysis and Training README](./analysis-and-training/README.md) for details.

### Model Storage and Management

See the [Model Server README](./model-server/README.md) for details.

### Making Predictions

See the [Inference Server README](./inference-server/README.md) for details.

## Running Tests

### Running service tests

To run all tests in sequence with a clean environment:

```bash
./run_all_tests.sh
```

### Component Tests

For running tests on individual components:

```bash
# Run model-server tests with clean environment
cd model-server/tests
./run_tests.sh

# Run inference-server tests with clean environment
cd inference-server/tests
./run_tests.sh
```

## API Documentation

OpenAPI/Swagger specifications are available for both servers:

- [Model Server API](./model-server/openapi.yaml)
- [Inference Server API](./inference-server/openapi.yaml)

## Component Documentation

For detailed documentation on each component, see:

- [Model Server](./model-server/README.md)
- [Inference Server](./inference-server/README.md)
- [Analysis and Training](./analysis-and-training/README.md)
- [Experiment Runner](./experiment-runner/README.md)
- [Shared Utilities](./shared/README.md)
- [Data Lake](./datalake/README.md)
