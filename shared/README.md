# Shared Utilities

Common utilities and modules used across multiple components in the PyTorch AI/ML Workflow project.

## ML Utilities (`ml_utils.py`)

Common machine learning functions and models used by both the model-server and inference-server components.

### Key Features

- **LinearRegressionModel**: Common PyTorch model definition with batch normalization
- **Model Training**: Functions for creating and training models with sample data
- **ONNX Export**: Utilities for exporting models to ONNX format
- **Input Processing**: Dynamic input shape detection and tensor formatting
- **Metadata Handling**: Metadata generation with configurability
- **Prediction Utilities**: Unified prediction handling

### Usage Example

```python
from shared.ml_utils import (
    create_and_train_model,
    export_model_to_onnx,
    get_default_metadata
)

# Create and train a model
model = create_and_train_model(input_features=2, output_features=1)

# Export to ONNX format
onnx_path = export_model_to_onnx(model, "/path/to/output.onnx")

# Generate metadata
metadata = get_default_metadata(
    model_name="my_model",
    version="1.0.0"
)
```

## Test Model Creation (`create_test_models.py`)

A script to create test models for both model-server and inference-server components.

### Usage

```bash
# Create all test models
./create_test_models.py

# With specific server URL
./create_test_models.py --url http://custom-server:5000

# Create only simple or versioned models
./create_test_models.py --simple
./create_test_models.py --versioned
```

## Test Utilities (`test_utils.py`)

Common utilities for testing both model-server and inference-server components.

### Key Features

- Model download and analysis functions
- Input data formatting based on model shape
- Server connectivity checks
- Model and metadata retrieval helpers

### Usage Example

```python
from shared.test_utils import (
    download_and_analyze_model,
    create_input_data_for_shape
)

# Get model and analyze its input shape
input_shape, _ = download_and_analyze_model(
    f"{MODEL_SERVER_URL}/models/uuid/{model_uuid}"
)

# Create test data based on the model shape
input_data = create_input_data_for_shape(input_shape)
```
