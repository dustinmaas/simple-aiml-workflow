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

# %%
from influxdb_client import InfluxDBClient

# Connect to InfluxDB
client = InfluxDBClient(url="http://metrics_influxdb:8086", token="ric_admin_token", org="ric")
experiment_id = "exp_1741030459"
query = '''
from(bucket: "network_metrics")
  |> range(start: 0, stop: now())
  |> filter(fn: (r) => r.experiment_id == "exp_1741030459")
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
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"device {device}")
device = 'cpu'

features = data[['CQI','DRB.UEThpDl']].values
targets = data[['min_prb_ratio']].values


# Convert features and targets to PyTorch tensors
X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(targets, dtype=torch.float32)

# %%
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)  # two input features, one output feature
        
        # Apply batch normalization to input features
        self.batch_norm = torch.nn.BatchNorm1d(2)
        
        # Register buffers to store the mean and standard deviation of the output features
        self.register_buffer('y_mean', torch.zeros(1))
        self.register_buffer('y_std', torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = self.batch_norm(x)
        output = self.linear(x_normalized)
        
        if not self.training:
            with torch.no_grad():
                output = output * self.y_std + self.y_mean
                
        return output

# %%
# Create and train the model
model = LinearRegressionModel()
model.y_mean = y.mean(dim=0, keepdim=True)
model.y_std = y.std(dim=0, keepdim=True)
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
# convert to onnx and save with versioning
import torch.onnx
import datetime
import json
import onnx

# Set the model to inference mode
model.eval()

# Model name and version info
model_name = "linear_regression_model"
model_version = "1.0.1"  # Follow semantic versioning: major.minor.patch

# Calculate loss as a metric
with torch.no_grad():
    y_pred = model(X)
    loss = criterion(y_pred, (y - model.y_mean) / model.y_std).item()

# Metadata to include with the model
metadata_props = {
    "version": model_version,
    "training_date": datetime.datetime.now().isoformat(),
    "framework": f"PyTorch {torch.__version__}",
    "dataset": "network_metrics_exp_1741030459",
    "metrics": json.dumps({"mse": loss}),
    "description": "Linear regression model for PRB prediction based on CQI and throughput",
    "input_features": json.dumps(["CQI", "DRB.UEThpDl"]),
    "output_features": json.dumps(["min_prb_ratio"])
}

# PyTorch needs to trace the operations inside the model to generate the computational graph
dummy_input = torch.randn(1, 2)  # Example dummy input for export

# Use a temp file for export before uploading to model server
temp_dir = tempfile.mkdtemp()
temp_model_path = os.path.join(temp_dir, f"{model_name}_v{model_version}.onnx")

# Export the model to ONNX without any embedded metadata
torch.onnx.export(
    model, 
    dummy_input, 
    temp_model_path, 
    verbose=True, 
    input_names=["input"], 
    output_names=["output"], 
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

# No longer adding metadata directly to the ONNX file
# It will be sent separately via the API

print(f"ONNX model saved at {temp_model_path} with version {model_version}")

# %%
# Upload model to model server with versioning
model_server_url = os.getenv("MODEL_SERVER_URL", "http://model-server:5000")

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
import onnxruntime as ort

model_name = "linear_regression_model"
model_version = "1.0.1"  # Specific version or use "/latest" to get the latest version

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
    
    # Load the ONNX model with onnxruntime for inference
    ort_session = ort.InferenceSession(temp_model_path)
    
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
    
    # Test inference with the model
    test_input = np.array([[10.0, 100.0]], dtype=np.float32)  # Example CQI and throughput
    outputs = ort_session.run(
        None,  # output names, None means return all outputs
        {"input": test_input}  # input data
    )
    print(f"\nTest inference:")
    print(f"Input (CQI, Throughput): {test_input}")
    print(f"Predicted min_prb_ratio: {outputs[0]}")
    
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
