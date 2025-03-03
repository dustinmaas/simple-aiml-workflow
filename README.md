# PyTorch AI/ML Workflow

This project demonstrates a complete AI/ML workflow using PyTorch for training and inference in a microservices architecture.

## Architecture

The project consists of the following components:

1. **Model Server** - Stores and serves trained PyTorch models
2. **Inference Server** - Provides an API for making predictions using the models
3. **Analysis and Training Environment** - Jupyter notebook environment for data analysis and model training
4. **Experiment Runner** - Tool for running end-to-end tests of the workflow

The services communicate through REST APIs and share a common library (`lib`) that provides:
- Custom `TorchStandardScaler` for standardization in PyTorch
- Common utilities and preprocessing functions

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
4. Run the experiment runner in test mode

### Running Components Individually

You can also run specific components:

```bash
# Start only the model and inference servers
docker-compose up model-server inference-server

# Start the Jupyter notebook environment
docker-compose up analysis-and-training

# Run experiment-runner in interactive mode
docker-compose run experiment-runner python /app/run_experiment.py --mode=interactive
```

## Project Structure

```
.
├── analysis-and-training/     # Jupyter notebooks for analysis and training
│   ├── notebooks/             # Python notebooks and scripts
│   └── Dockerfile
├── model-server/              # Model storage and serving
│   ├── app.py                 # Flask app for model management
│   └── Dockerfile
├── inference-server/          # Inference API service
│   ├── app.py                 # Flask app for making predictions
│   └── Dockerfile
├── experiment-runner/         # Tools for running experiments
│   ├── run_experiment.py      # Main experiment runner
│   ├── test_torch_inference.py # End-to-end test script
│   └── Dockerfile
├── lib/                       # Shared library code
│   └── torch_scaler.py        # PyTorch-based StandardScaler
└── docker-compose.yml         # Docker Compose configuration
```

## Using the Project

### Training a Model

1. Access the Jupyter notebook at http://localhost:8888
2. Open `analysis-and-training/notebooks/train_torch_model_notebook.ipynb`
3. Run the notebook to:
   - Generate synthetic data or load your own data
   - Train a PyTorch model using the `TorchStandardScaler`
   - Save the model to the model server

### Making Predictions

Send a POST request to the inference server:

```bash
curl -X POST http://localhost:5002/predict/prb \
  -H "Content-Type: application/json" \
  -d '{"cqi": 10, "throughput": 80, "model_id": "torch_linear_regression_v1"}'
```

### Running Tests

You can run the end-to-end test script:

```bash
docker-compose run experiment-runner python /app/test_torch_inference.py
```

Or modify the Docker Compose configuration to run in test mode automatically:

```yaml
command: python /app/run_experiment.py --mode=test
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 