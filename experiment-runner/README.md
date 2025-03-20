# Experiment Runner

A tool for running end-to-end experiments to generate datasets and orchestrate the AI/ML workflow.

## Features

- YAML-based configuration for flexible experiment setup
- Data generation through controlled network experiments
- Integration with InfluxDB for metrics storage
- Communication with Model Server and Inference Server
- Support for both data generation and inference modes
- Experiment orchestration and automation

## Configuration

The Experiment Runner uses YAML configuration files located in the `config/` directory:

- `default.yaml`: Standard configuration with extended experiment parameters
- `test.yaml`: Shorter test configuration for quick validation
- `node_config.yaml`: Configuration for node controllers and connections

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
