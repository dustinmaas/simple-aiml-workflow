# Experiment Runner

A tool for running end-to-end experiments to generate datasets and orchestrate the AI/ML workflow.

## Features

- Data generation through controlled network experiments
- Experiment orchestration and automation with YAML-based configuration for experiment parameters
- Export of metrics to Data Lake (InfluxDB)
- Communication with Model Server and Inference Server
- Support for both data generation and inference modes (WIP)

## Configuration

The Experiment Runner uses YAML configuration files located in the `config/` directory:

- `default.yaml`: Standard configuration with extended experiment parameters
- `test.yaml`: Shorter test configuration for quick validation
- `node_config.yaml`: Configuration for node controllers and connections
  - This file will depend on the powder experiment setup and the experiments you'd like to run

## Usage

To run an experiment:

```bash
cd experiment-runner
./runner --config=default
```

For a shorter test experiment:

```bash
cd experiment-runner
./runner --config=test
```

## Workflow

1. Start required services (InfluxDB, O-RAN SC RIC)
2. Start core network and establish connections
3. Configure the experiment parameters according to YAML config
4. Iterate through configured test cases, collecting data
5. Store metrics in InfluxDB for later analysis

## Future Enhancements

A high-priority enhancement is adding inference mode, which will:
- Retrieve models from the Model Server
- Use the Inference Server to make predictions
- Apply prediction results to network configurations
- Create a closed-loop optimization system
