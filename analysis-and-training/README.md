# Analysis and Training Environment

A Jupyter-based environment for data analysis, model development, training, and exporting to the Model Server.

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

Note: These scripts are actually Jupyter notebooks encoded in the [jupytext](https://jupytext.readthedocs.io/en/latest/) format for better compatability with version control. They can be converted to normal Jupyter notebooks (ipynb) for interactive use, or used directly.

## Workflow Example

1. Load data from the data lake (InfluxDB)
1. Perform data analysis and visualization
1. Process the data to generate a dataset for model training
1. Develop and train a PyTorch model (linear and polynomial regression examples)
1. Export the trained model to ONNX format
1. Generate metadata for the model
1. Upload the model and metadata to the Model Server
1. Alternatively, upload to HuggingFace or another cloud-based model storage service

## Environment Variables

- `JUPYTER_IP`: Host to bind Jupyter server (default: 0.0.0.0)
- `JUPYTER_PORT`: Port to run Jupyter server (default: 8888)
- `JUPYTER_TOKEN`: Token for Jupyter server authentication (default: "somejupytertoken")
