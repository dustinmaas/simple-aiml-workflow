# Analysis and Training Environment

A Jupyter-based environment for data analysis, model development, training, and exporting to the Model Server.

## Features

- Jupyter Notebook server for interactive development
- PyTorch for model development and training
- Tools for data analysis and visualization
- ONNX export utilities for model interoperability
- Integration with both Model Server and InfluxDB
- Cloud-based model storage options (Hugging Face)

## Usage

### Starting the Environment

```bash
sudo docker compose up -d analysis-and-training
```

### Accessing Jupyter

Once running, access the Jupyter environment at:

```
http://localhost:8888
```

## Key Notebooks and Scripts

- **playground.py**: Basic model development with PyTorch and export to ONNX
- **cyberpowder.py**: Example of cloud-based model storage using Hugging Face
- **cyberpowder-dataset-analysis.py**: Example of data analysis with visualization

## Workflow Example

1. Load and analyze data from InfluxDB or CSV files
2. Develop and train a PyTorch model (linear or polynomial regression)
3. Export the trained model to ONNX format
4. Generate metadata for the model
5. Upload the model and metadata to the Model Server
6. Alternatively, upload to Hugging Face for cloud-based storage

## Environment Variables

- `JUPYTER_IP`: Host to bind Jupyter server (default: 0.0.0.0)
- `JUPYTER_PORT`: Port to run Jupyter server (default: 8888)
