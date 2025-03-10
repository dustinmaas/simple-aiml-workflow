# Shared Utilities

This directory contains shared utilities and modules used across multiple components in the PyTorch AI/ML Workflow project.

## ML Utilities

The `ml_utils.py` module provides common machine learning functions and models used by both the model-server and inference-server components.

### Features

- **LinearRegressionModel**: A common PyTorch model definition with batch normalization and shape reporting
- **Model Training**: Configurable functions for creating and training models with sample data
- **ONNX Export**: Enhanced utilities for exporting models to ONNX format with validation
- **ONNX Session Management**: Version-compatible session creation and in-memory model handling
- **Input Processing**: Dynamic input shape detection and tensor formatting
- **Metadata Handling**: Comprehensive metadata generation with additional configurability
- **Prediction Utilities**: Unified prediction handling for both file and in-memory models

### Cross-Container Compatibility

The shared utilities are designed to work seamlessly across different container environments, with robust import handling that adapts to various execution contexts:

```python
# Dynamic module resolution
try:
    # When imported as a module
    from .ml_utils import create_and_train_model
except ImportError:
    # When run as a script
    from ml_utils import create_and_train_model
```

This approach ensures that the shared utilities can be imported correctly whether they're:
- Running inside Docker containers
- Executed as standalone scripts
- Imported as modules from other components
- Called from test environments

### Usage

```python
# Import with flexibility for different contexts
try:
    from shared.ml_utils import (
        create_and_train_model,
        export_model_to_onnx,
        get_default_metadata,
        create_onnx_session,
        run_prediction
    )
except ImportError:
    import sys
    import os
    # Add parent directory to path if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from shared.ml_utils import (
        create_and_train_model,
        export_model_to_onnx,
        get_default_metadata,
        create_onnx_session,
        run_prediction
    )

# Create and train a model with optional parameters
model = create_and_train_model(
    input_features=2,
    output_features=1,
    num_epochs=100
)

# Export to ONNX format with verification
onnx_path = export_model_to_onnx(
    model, 
    "/path/to/output.onnx",
    input_names=["input"],
    output_names=["output"]
)

# Generate comprehensive metadata
metadata = get_default_metadata(
    model_name="my_model",
    version="1.0.0",
    description="My model description",
    input_features=["feature1", "feature2"],
    output_features=["prediction"]
)

# Create a version-compatible ONNX session
session = create_onnx_session(onnx_path)

# Run prediction with the model
result = run_prediction(onnx_path, {"input": [1.0, 2.0]})
```

## Test Model Creation

The `create_test_models.py` script provides a unified way to create test models for both model-server and inference-server components.

### Features

- **Command Line Interface**: Flexible CLI with parameter configuration
- **Environment Variable Integration**: Automatic detection of model server URL
- **Selective Model Creation**: Options to create only specific test models
- **Improved Error Handling**: Enhanced exception handling and reporting
- **Reusable Components**: Functions that can be imported or run as a script

### Usage

```bash
# Create all test models with default server URL
./create_test_models.py

# Create models with a specific server URL
./create_test_models.py --url http://custom-server:5000

# Create only a simple test model (not versioned models)
./create_test_models.py --simple

# Create only versioned test models
./create_test_models.py --versioned
```

Or import as a module:

```python
from shared.create_test_models import create_test_model, create_versioned_test_models

# Create a single test model
model_info = create_test_model("http://localhost:5000")

# Create multiple versioned models
versioned_models = create_versioned_test_models("http://localhost:5000")
```
