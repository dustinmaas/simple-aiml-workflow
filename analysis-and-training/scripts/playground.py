# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import torch
import numpy as np
import os
import tempfile
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import sys
from pathlib import Path
import io
import json
import datetime
import onnx
import onnxruntime as ort
from typing import Dict, List, Any, Optional, Tuple, Union, BinaryIO

# %%
from influxdb_client import InfluxDBClient

# Connect to InfluxDB using environment variables
client = InfluxDBClient(
    url=f"http://{os.getenv('INFLUXDB_HOST')}:{os.getenv('INFLUXDB_PORT')}", 
    token=os.getenv("INFLUXDB_ADMIN_TOKEN"), 
    org=os.getenv("INFLUXDB_ORG")
)
experiment_id = "exp_1741030459"
# Get InfluxDB bucket from environment
influx_bucket = os.getenv("INFLUXDB_BUCKET", "network_metrics")

query = f'''
from(bucket: "{influx_bucket}")
  |> range(start: 0, stop: now())
  |> filter(fn: (r) => r.experiment_id == "{experiment_id}")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["timestamp", "ue_id", "atten", "min_prb_ratio", "CQI", "RSRP", "DRB.UEThpDl", "DRB.RlcSduTransmittedVolumeDL"])
'''

result = client.query_api().query_data_frame(query=query)


# Convert columns to appropriate data types
result['ue_id'] = result['ue_id'].astype(int)
result['atten'] = result['atten'].astype(int) 
result['min_prb_ratio'] = result['min_prb_ratio'].astype(int)
result['CQI'] = result['CQI'].astype(int)
result['RSRP'] = result['RSRP'].astype(int)
result['DRB.UEThpDl'] = result['DRB.UEThpDl'].astype(float)
result['DRB.RlcSduTransmittedVolumeDL'] = result['DRB.RlcSduTransmittedVolumeDL'].astype(float)
result['timestamp'] = pd.to_datetime(result['timestamp'])

# Convert to pandas DataFrame
df = pd.DataFrame(result)
df

# Drop InfluxDB metadata columns
df.drop(columns=['result', 'table'], inplace=True)
df


# %%
# in Mbps
df['DRB.UEThpDl'] = df['DRB.UEThpDl'] / 1000.0
df['DRB.RlcSduTransmittedVolumeDL'] = df['DRB.RlcSduTransmittedVolumeDL'] / 1000.0
df.describe()
df



# %%

# grab ue1 data
ue1_df = df[df['ue_id'] == 1].copy()

# fix default min_prb_ratio at start (better to fix in experiment runner
# ue1_df['min_prb_ratio'] = ue1_df['min_prb_ratio'].replace(0, 50)
print(ue1_df.dtypes)
ue1_df

# %%
ue1_df.describe()

# %%
# working on filtering out transient vals (not done)
# ue1_df['min_thp_per'] = df.groupby('min_prb_ratio')['DRB.UEThpDl'].transform('min')
# ue1_df[ue1_df['min_prb_ratio'] == 50]
#ue1_df[ue1_df['DRB.UEThpDl'] == ue1_df['min_thp_per']].to_string()

# %%
# get a general idea of what the relevant data points look like
# Create a Plotly time series figure for all metrics
fig = go.Figure()

# Add each metric as a separate trace
fig.add_trace(go.Scatter(x=ue1_df['timestamp'], y=ue1_df['atten'], mode='lines', name='atten'))
fig.add_trace(go.Scatter(x=ue1_df['timestamp'], y=ue1_df['CQI'], mode='lines', name='CQI'))
fig.add_trace(go.Scatter(x=ue1_df['timestamp'], y=ue1_df['RSRP'], mode='lines', name='RSRP'))
fig.add_trace(go.Scatter(x=ue1_df['timestamp'], y=ue1_df['DRB.UEThpDl'], mode='lines', name='DRB.UEThpDl (Mbps)'))
fig.add_trace(go.Scatter(x=ue1_df['timestamp'], y=ue1_df['min_prb_ratio'], mode='lines', name='min_prb_ratio'))

# Update layout
fig.update_layout(
    title='Time Series of Network Metrics',
    xaxis_title='Timestamp',
    yaxis_title='Value',
    legend_title='Metrics',
    hovermode='x unified'
)

# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

fig.show()

# %%
# Tput vs. CQI and min prb ratio
# note the transient values at the ratio switch points (95 -> 50 is the most egregious) 
# Create a function to make a scatter plot for a specific min_prb_ratio value
def make_scatter_for_prb(df, prb_value):
    df_filtered = df[df['min_prb_ratio'] == prb_value]
    return go.Scatter(
        x=df_filtered['CQI'],
        y=df_filtered['DRB.UEThpDl'],
        mode='markers',
        name=f'min_prb_ratio = {prb_value}',
        marker=dict(
            size=8,
            opacity=0.7,
        ),
        hovertemplate='CQI: %{x}<br>Throughput: %{y:.2f} Mbps<extra></extra>'
    )

# Get unique min_prb_ratio values
unique_prb_values = sorted(ue1_df['min_prb_ratio'].unique())

# Create subplot grid with one subplot per min_prb_ratio value
fig = make_subplots(
    rows=1, 
    cols=len(unique_prb_values),
    subplot_titles=[f'min_prb_ratio = {val}' for val in unique_prb_values],
    shared_yaxes=True
)

# Add a scatter trace for each min_prb_ratio value
for i, prb_value in enumerate(unique_prb_values):
    fig.add_trace(
        make_scatter_for_prb(ue1_df, prb_value),
        row=1, 
        col=i+1
    )

# Update layout
fig.update_layout(
    title='Throughput vs. CQI by min_prb_ratio',
    height=500,
    width=200 * len(unique_prb_values),
    showlegend=False
)

# Update axes labels
for i in range(len(unique_prb_values)):
    fig.update_xaxes(title_text="CQI", row=1, col=i+1)
    if i == 0:  # Only add y-axis title to the first subplot
        fig.update_yaxes(title_text="Throughput (Mbps)", row=1, col=i+1)

fig.show()

# %%
# another view
data = ue1_df[['CQI','DRB.UEThpDl', 'min_prb_ratio']]

# Create a scatter matrix with Plotly
# Use Plotly Express for pairplot equivalent
fig = px.scatter_matrix(
    data,
    dimensions=["CQI", "DRB.UEThpDl", "min_prb_ratio"],
    color="DRB.UEThpDl",
    color_continuous_scale=px.colors.sequential.Viridis,
    opacity=0.8,
    title="Scatter Matrix (Pair Plot) of Network Metrics"
)

# Update layout
fig.update_layout(
    width=800,
    height=800,
    plot_bgcolor='white'
)

# Update traces
fig.update_traces(
    diagonal_visible=False,
    showupperhalf=False,
    marker=dict(size=5)
)

fig.show()

# %%
class LinearRegressionModel(torch.nn.Module):
    """
    Linear regression model with batch normalization.
    
    This model is designed to predict min_prb_ratio based on input features
    like CQI and throughput. It includes batch normalization for inputs and
    stores normalization parameters for outputs to ensure consistent predictions.
    """
    def __init__(self, input_features: int = 2, output_features: int = 1):
        """
        Initialize the model with configurable feature dimensions.
        
        Args:
            input_features: Number of input features (default: 2)
            output_features: Number of output features (default: 1)
        """
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_features, output_features)
        
        # Apply batch normalization to input features
        self.batch_norm = torch.nn.BatchNorm1d(input_features)
        
        # Register buffers to store the mean and standard deviation of the output features
        self.register_buffer('y_mean', torch.zeros(output_features))
        self.register_buffer('y_std', torch.ones(output_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor with shape [batch_size, input_features]
            
        Returns:
            Output tensor with shape [batch_size, output_features]
        """
        x_normalized = self.batch_norm(x)
        output = self.linear(x_normalized)
        
        # Denormalize output during inference
        if not self.training:
            with torch.no_grad():
                output = output * self.y_std + self.y_mean
                
        return output
    
    def get_input_shape(self) -> List[int]:
        """
        Get the expected input shape for this model.
        
        Returns:
            List representing the input shape [batch_size, input_features]
        """
        return [None, self.batch_norm.num_features]
    
    def get_output_shape(self) -> List[int]:
        """
        Get the expected output shape for this model.
        
        Returns:
            List representing the output shape [batch_size, output_features]
        """
        return [None, self.linear.out_features]



# %%
# Create tensors for features and target
X = torch.tensor(ue1_df[['CQI', 'DRB.UEThpDl']].values, dtype=torch.float32)
y = torch.tensor(ue1_df['min_prb_ratio'].values, dtype=torch.float32).reshape(-1, 1)

# Create and train the model using the locally defined model class
model = LinearRegressionModel(input_features=2, output_features=1)
model.y_mean = y.mean(dim=0, keepdim=True)
model.y_std = y.std(dim=0, keepdim=True)

device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
X.to(device)
y.to(device)
criterion = torch.nn.MSELoss() # Mean Squared Error
optimizer = torch.optim.SGD(model.parameters(), lr=.05)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    # Forward pass
    y_predicted = model(X)
    loss = criterion(y_predicted, (y - model.y_mean) / model.y_std)
    # Backward and optimize
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# %% 
# Define some helper functions for model export and inference
def export_model_to_onnx(
    model: torch.nn.Module, 
    file_path: str, 
    input_shape: Optional[List[int]] = None,
    input_names: List[str] = ["input"], 
    output_names: List[str] = ["output"],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
) -> str:
    """
    Export PyTorch model to ONNX format with configurable parameters.
    
    Args:
        model: PyTorch model to export
        file_path: Path where the ONNX model will be saved
        input_shape: Shape of the dummy input (default: [1, 2])
        input_names: Names for the input tensors (default: ["input"])
        output_names: Names for the output tensors (default: ["output"])
        dynamic_axes: Dictionary specifying dynamic axes (default: batch_size is dynamic)
        
    Returns:
        Path to the exported ONNX model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Use default input shape if not provided
    if input_shape is None:
        if hasattr(model, 'get_input_shape'):
            # Try to get shape from model if it has the method
            shape = model.get_input_shape()
            # Replace None with 1 for batch dimension
            input_shape = [1 if dim is None else dim for dim in shape]
        else:
            # Default to [1, 2] for LinearRegressionModel
            input_shape = [1, 2]
    
    # Create dummy input with the specified shape
    dummy_input = torch.randn(*input_shape)
    
    # Set up dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"}, 
            "output": {0: "batch_size"}
        }
    
    # Set model to inference mode
    model.eval()
    
    # Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        file_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    return file_path

def get_default_metadata(
    model_name: Optional[str] = None, 
    version: Optional[str] = None, 
    description: Optional[str] = None,
    input_features: Optional[List[str]] = None,
    output_features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create default metadata for a model with configurable fields.
    
    Args:
        model_name: Optional name of the model
        version: Optional version string
        description: Optional description of the model
        input_features: Optional list of input feature names
        output_features: Optional list of output feature names
        
    Returns:
        Dictionary with model metadata
    """
    # Default input and output features if not provided
    if input_features is None:
        input_features = ["CQI", "DRB.UEThpDl"]
    
    if output_features is None:
        output_features = ["min_prb_ratio"]
    
    metadata = {
        'description': description or 'Linear regression model for PRB prediction based on CQI and throughput',
        'training_date': datetime.datetime.now().isoformat(),
        'input_features': json.dumps(input_features),
        'output_features': json.dumps(output_features),
        'framework': f'PyTorch {torch.__version__}'
    }
    
    # Add optional fields if provided
    if model_name:
        metadata['model_name'] = model_name
    
    if version:
        metadata['version'] = version
        
    return metadata

def create_onnx_session(model_path: str) -> ort.InferenceSession:
    """
    Create an ONNX inference session with version compatibility handling.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        ONNX runtime inference session
    """
    session_options = ort.SessionOptions()
    providers = ['CPUExecutionProvider']
    
    try:
        # For newer versions of ONNX Runtime, providers parameter is a keyword arg
        return ort.InferenceSession(model_path, providers=providers, sess_options=session_options)
    except TypeError:
        # Fallback for older versions of ONNX Runtime where providers was a positional arg
        return ort.InferenceSession(model_path, session_options, providers)

def format_input_tensor(tensor: np.ndarray, input_shape: Optional[List[Optional[int]]]) -> np.ndarray:
    """
    Format input tensor to match expected model input shape.
    
    Args:
        tensor: Input tensor as numpy array
        input_shape: Expected input shape from model
        
    Returns:
        Properly formatted tensor
    """
    # If no shape provided or shape is all None, do minimal formatting
    if input_shape is None or all(dim is None for dim in input_shape):
        # Handle reshaping if needed (e.g., if tensor is 1D, make it 2D)
        if len(tensor.shape) == 1:
            return tensor.reshape(1, -1)
        return tensor
    
    # For 2D inputs like [batch_size, features], ensure input has the right shape
    if len(input_shape) == 2 and len(tensor.shape) == 1:
        # Convert [x] to [[x]] - reshape 1D to 2D (with batch size 1)
        return tensor.reshape(1, -1)
    
    # Add batch dimension if needed
    if len(input_shape) > len(tensor.shape):
        # If tensor dimensions are fewer than model expects, add batch dimension
        return tensor.reshape(1, *tensor.shape)
    
    return tensor

def run_prediction(
    model_path_or_session: Union[str, bytes, ort.InferenceSession], 
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run inference with an ONNX model.
    
    Args:
        model_path_or_session: Path to the ONNX model file, bytes of model, or InferenceSession
        input_data: Input data for the model
        
    Returns:
        Prediction results
    """
    # Create session if needed
    if isinstance(model_path_or_session, str):
        session = create_onnx_session(model_path_or_session)
    elif isinstance(model_path_or_session, bytes):
        # Create session from buffer (not used in this script but included for compatibility)
        session_options = ort.SessionOptions()
        providers = ['CPUExecutionProvider']
        try:
            session = ort.InferenceSession(model_path_or_session, providers=providers, sess_options=session_options)
        except TypeError:
            session = ort.InferenceSession(model_path_or_session, session_options, providers)
    else:
        session = model_path_or_session
    
    # Get input and output names
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    
    # Prepare input tensors
    input_tensors = {}
    for name in input_names:
        if name in input_data:
            # Convert input data to numpy array with explicit float32 type
            # This ensures compatibility with models expecting float32 tensors
            tensor = np.array(input_data[name], dtype=np.float32)
            
            # Get expected shape from input definition
            input_shape = None
            for model_input in session.get_inputs():
                if model_input.name == name:
                    input_shape = model_input.shape
                    break
            
            # Format tensor using helper function
            tensor = format_input_tensor(tensor, input_shape)
            
            input_tensors[name] = tensor
        else:
            raise ValueError(f"Input {name} not found in request data")
    
    # Run inference
    outputs = session.run(output_names, input_tensors)
    
    # Format results
    result = {}
    for i, name in enumerate(output_names):
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(outputs[i], np.ndarray):
            result[name] = outputs[i].tolist()
        else:
            result[name] = outputs[i]
    
    return result

# %%

# Model name and version info
model_name = "linear_regression_model"
model_version = "1.0.2"  # Follow semantic versioning: major.minor.patch

# Calculate loss as a metric
with torch.no_grad():
    y_pred = model(X)
    loss = criterion(y_pred, (y - model.y_mean) / model.y_std).item()

# Get default metadata and enhance it with additional information
metadata_props = get_default_metadata(
    model_name=model_name,
    version=model_version,
    description="Linear regression model for PRB prediction based on CQI and throughput",
    input_features=["CQI", "DRB.UEThpDl"],
    output_features=["min_prb_ratio"]
)

# Add additional custom metadata
metadata_props.update({
    "dataset": "network_metrics_exp_1741030459",
    "metrics": json.dumps({"mse": loss})
})

# Use a temp file for export before uploading to model server
temp_dir = tempfile.mkdtemp()
temp_model_path = os.path.join(temp_dir, f"{model_name}_v{model_version}.onnx")

# Export the model to ONNX using the locally defined utility function
export_model_to_onnx(
    model=model,
    file_path=temp_model_path,
    input_names=["input"],
    output_names=["output"]
)

print(f"ONNX model saved at {temp_model_path} with version {model_version}")

# %%
# Upload model to model server with versioning - using environment variables
model_server_url = f"http://{os.getenv('MODEL_SERVER_HOST')}:{os.getenv('MODEL_SERVER_PORT')}"

# Send to model server via REST API using the versioned endpoint with metadata
with open(temp_model_path, 'rb') as f:
    files = {'model': f}
    form_data = {'metadata': json.dumps(metadata_props)}
    upload_url = f"{model_server_url}/models/{model_name}/versions/{model_version}"
    response = requests.post(upload_url, files=files, data=form_data)
    
if response.status_code == 200:
    print(f"Model {model_name} version {model_version} successfully uploaded to model server")
    print(f"Response: {response.json()}")
elif response.status_code == 409:
    print(f"Version {model_version} already exists. Consider incrementing the version number.")
    # Alternatively, could auto-increment the patch version here
else:
    print(f"Error uploading model: {response.text}")
    
# Cleanup temp file
os.remove(temp_model_path)
os.rmdir(temp_dir)

# %%
# Get model from model server using versioning APIs

model_name = "linear_regression_model"
model_version = "1.0.2"  # Specific version or use "/latest" to get the latest version

# Option 1: Get specific version
response = requests.get(f'{model_server_url}/models/{model_name}/versions/{model_version}')

# Option 2: Get latest version
# response = requests.get(f'{model_server_url}/models/{model_name}/versions/latest')

if response.status_code == 200:
    # Create temp file to save the downloaded model
    temp_dir = tempfile.mkdtemp()
    temp_model_path = os.path.join(temp_dir, f"{model_name}_v{model_version}.onnx")
    
    # Save the model to the temp file
    with open(temp_model_path, 'wb') as f:
        f.write(response.content)
    
    # Load the ONNX model with onnxruntime for inference using the locally defined utility function
    ort_session = create_onnx_session(temp_model_path)
    
    # Get metadata from the metadata endpoint
    metadata_response = requests.get(f'{model_server_url}/models/{model_name}/versions/{model_version}/metadata')
    if metadata_response.status_code == 200:
        metadata = metadata_response.json()
        print("Metadata retrieved from separate endpoint")
    else:
        # Fall back to embedded metadata if available
        print("Metadata endpoint not available, trying embedded metadata")
        metadata = {prop.key: prop.value for prop in ort_session.get_modelmeta().custom_metadata_map.items()}
    
    # Print model info
    print(f"Model: {model_name} (version {model_version})")
    print(f"Training date: {metadata.get('training_date', 'unknown')}")
    print(f"Metrics: {metadata.get('metrics', 'unknown')}")
    print(f"Description: {metadata.get('description', 'unknown')}")
    
    # Test inference with the model using the locally defined utility function
    test_input = np.array([[10.0, 100.0]], dtype=np.float32)  # Example CQI and throughput
    
    # Using the run_prediction function
    prediction_result = run_prediction(
        ort_session,
        {"input": test_input}
    )
    
    print(f"\nTest inference:")
    print(f"Input (CQI, Throughput): {test_input}")
    print(f"Predicted min_prb_ratio: {prediction_result['output']}")
    
    # Cleanup
    del ort_session
    os.remove(temp_model_path)
    os.rmdir(temp_dir)
    
    print("\nModel successfully loaded from model server")
else:
    print(f"Error downloading model: {response.text}")

# %%
# look at the hyperplane fit

# Get the learned parameters
learned_weights = model.linear.weight.data.cpu().numpy()
learned_bias = model.linear.bias.data.cpu().numpy()

# Print the learned hyperplane equation (in scaled space)
print("\nLearned Hyperplane (in scaled space):")
print(f"y_scaled = {learned_weights[0][0]:.2f}*x1_scaled + {learned_weights[0][1]:.2f}*x2_scaled + {learned_bias[0]:.2f}")

# Get feature normalization parameters from the batch normalization layer
x_mean = model.batch_norm.running_mean.cpu().numpy().reshape(1, -1)
x_std = torch.sqrt(model.batch_norm.running_var).cpu().numpy().reshape(1, -1)

# Create scaled versions of features and targets using the model's normalization parameters
features_scaled = (X.cpu().numpy() - x_mean) / x_std
targets_scaled = (y.cpu().numpy() - model.y_mean.cpu().numpy()) / model.y_std.cpu().numpy()

# Get original unscaled data
features_unscaled = X.cpu().numpy()
targets_unscaled = y.cpu().numpy()

# Sample every 5th point for clarity in the scatter plot
sample_indices = np.arange(0, len(features_unscaled), 5)
features_sampled = features_unscaled[sample_indices]
targets_sampled = targets_unscaled[sample_indices]

# Create a meshgrid for the hyperplane using unscaled feature ranges
x1_range = np.linspace(features_unscaled[:,0].min(), features_unscaled[:,0].max(), 20)
x2_range = np.linspace(features_unscaled[:,1].min(), features_unscaled[:,1].max(), 20)
X1, X2 = np.meshgrid(x1_range, x2_range)

# Convert meshgrid to scaled space for prediction using the batch normalization parameters
X1_scaled = (X1 - x_mean[0, 0]) / x_std[0, 0]
X2_scaled = (X2 - x_mean[0, 1]) / x_std[0, 1]

# Calculate predictions in scaled space
Y_predicted_scaled = learned_weights[0][0] * X1_scaled + learned_weights[0][1] * X2_scaled + learned_bias[0]

# Convert predictions back to unscaled space
Y_predicted_unscaled = Y_predicted_scaled * model.y_std.cpu().numpy()[0, 0] + model.y_mean.cpu().numpy()[0, 0]

# Create Plotly figure
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

# Add scatter plot for data points
scatter = go.Scatter3d(
    x=features_sampled[:,0],
    y=features_sampled[:,1],
    z=targets_sampled.flatten(),
    mode='markers',
    marker=dict(
        size=2,
        color='red',
        opacity=0.8
    ),
    name='Data Points',
    hovertemplate='CQI: %{x:.2f}<br>Throughput: %{y:.2f} Mbps<br>min_prb_ratio: %{z:.2f}<extra></extra>'
)
fig.add_trace(scatter)

# Add surface plot for the hyperplane
surface = go.Surface(
    x=X1, 
    y=X2, 
    z=Y_predicted_unscaled,
    colorscale='Blues',
    opacity=0.7,
    showscale=False,
    name='Predicted Hyperplane',
    hovertemplate='CQI: %{x:.2f}<br>Throughput: %{y:.2f} Mbps<br>Predicted min_prb_ratio: %{z:.2f}<extra></extra>'
)
fig.add_trace(surface)

# Update layout with labels and title
fig.update_layout(
    title='Hyperplane Fit (Unscaled Values)',
    scene=dict(
        xaxis_title='CQI',
        yaxis_title='DRB.UEThpDL (Mbps)',
        zaxis_title='min_prb_ratio',
        aspectmode='auto'
    ),
    legend=dict(
        y=0.99,
        x=0.01,
        font=dict(size=12)
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    width=800,
    height=600
)

# Show the interactive plot
fig.show()

# %%
