# AIML Workflow for O-RAN SC RIC

This project provides a complete AIML workflow for the O-RAN SC RIC platform, including data collection, model training, and inference.

## Components

The project consists of the following components:

1. **Experiment Runner**: Collects data from the RIC platform and runs experiments.
2. **Analysis and Training Environment**: Jupyter Lab environment for data analysis and model training.
3. **Model Server**: Stores and serves PyTorch models.
4. **Inference Server**: Provides inference API for models.

All components use [uv](https://astral.sh/uv) for fast Python package installation.

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Access to the O-RAN SC RIC platform
- SSH access to the UE, gNB, and CN nodes

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/simple-aiml-workflow.git
   cd simple-aiml-workflow
   ```

2. Start the services:
   ```bash
   docker compose up -d
   ```

   This will start the model-server, inference-server, and analysis-and-training services.

3. Access the Jupyter Lab environment:
   Open your browser and navigate to `http://localhost:8889`

### Running Experiments

To run an experiment:

```bash
docker compose run experiment-runner --config=test
```

## Architecture

![Architecture Diagram](docs/architecture.png)

- **Experiment Runner**: Collects data from the RIC platform and stores it in InfluxDB.
- **Analysis and Training Environment**: Provides a Jupyter Lab environment for data analysis and model training.
- **Model Server**: Stores and serves PyTorch models.
- **Inference Server**: Provides inference API for models.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 