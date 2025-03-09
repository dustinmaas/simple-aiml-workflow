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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import datasets
import huggingface_hub as hf
import onnx
import onnxruntime as ort

# Get Hugging Face token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("Warning: HF_TOKEN environment variable not found. You may need to set it for Hugging Face operations.")

# %%
df = pd.read_csv('/app/data/data.csv')

# %%
# in Mbps
# df['DRB.UEThpDl'] = df['DRB.UEThpDl'] / 1000.0
# df['DRB.RlcSduTransmittedVolumeDL'] = df['DRB.RlcSduTransmittedVolumeDL'] / 1000.0
# df.describe()
# df

# %%
# use datasets to create a dataset and upload to huggingface

# Convert the pandas DataFrame to a Hugging Face Dataset
dataset = datasets.Dataset.from_pandas(df)

# Display basic information about the dataset
print(f"Dataset has {len(dataset)} rows and the following features:")
print(dataset.features)

# Create a sample split to test the dataset
# dataset_dict = dataset.train_test_split(test_size=0.2, seed=42)
# print(f"Train set: {len(dataset_dict['train'])} examples")
# print(f"Test set: {len(dataset_dict['test'])} examples")

# %%
# Push the dataset to the Hugging Face Hub
# First, login to Hugging Face (you'll need to provide a token)
# To generate a token, go to: https://huggingface.co/settings/tokens

# Uncomment and run the login line when you're ready to upload


# %%
# Set the repository name for your dataset
repo_name = "cyberpowder/cyberpowder-network-metrics"  # Replace with your username

hf_api = hf.HfApi()
hf_api.create_repo(repo_name, private=True, token=HF_TOKEN)


# %%

# Uncomment and run the following lines to push the dataset to the Hub
dataset.push_to_hub(
    repo_name,
    private=True,  # Set to False to make it publicly accessible
    token=HF_TOKEN,
)

# print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_name}")

# %%
# load the dataset from the hub
dataset = datasets.load_dataset(repo_name, token=HF_TOKEN)

# %%
# Create a new configuration with selected features and train-test split

# Define features to use as model inputs and targets

df = dataset['train'].to_pandas()

# Filter for ue_id 1 only
print(f"Original dataset size: {len(df)} rows")
df = df[df['ue_id'] == 1]
print(f"After filtering for ue_id 1: {len(df)} rows")

# Create a new dataset from the processed DataFrame
ml_dataset = datasets.Dataset.from_pandas(df)

# # Create train-test-validation splits (70/20/10)
# splits = ml_dataset.train_test_split(test_size=0.3, seed=42)
# test_valid = splits['test'].train_test_split(test_size=0.33, seed=42)
    

# %%
# Push the updated dataset with the new ML configuration to HuggingFace
ml_dataset.push_to_hub(
    repo_name,
    config_name="ue1_ml_ready",  # This creates a new configuration in the same repo
    private=True,
    token=HF_TOKEN,
)

print(f"UE1 ML-ready dataset configuration successfully pushed to {repo_name}")
print(f"To load this specific configuration: datasets.load_dataset('{repo_name}', 'ue1_ml_ready')")

# %%

dataset_ue1 = datasets.load_dataset(repo_name, 'ue1_ml_ready', token=HF_TOKEN)

# %%
df = dataset_ue1['train'].to_pandas()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.describe()

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
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['atten'], mode='lines', name='atten'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['CQI'], mode='lines', name='CQI'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSRP'], mode='lines', name='RSRP'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['DRB.UEThpDl'] / 1000.0, mode='lines', name='DRB.UEThpDl (Mbps)'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['min_prb_ratio'], mode='lines', name='min_prb_ratio'))

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
unique_prb_values = sorted(df['min_prb_ratio'].unique())

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
        make_scatter_for_prb(df, prb_value),
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
data = df[['CQI','DRB.UEThpDl', 'min_prb_ratio']]

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

dataset_ue1.set_format(type='torch', columns=['CQI', 'DRB.UEThpDl', 'min_prb_ratio'], dtype=torch.float32)

# Convert features and targets to PyTorch tensors
X = torch.stack([dataset_ue1['train']['CQI'], dataset_ue1['train']['DRB.UEThpDl']], dim=1)
y = dataset_ue1['train']['min_prb_ratio'].unsqueeze(-1)

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
num_epochs = 150
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
# Save the model to a temporary file
import datetime
import json

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

# Send the model and metadata to Hugging Face
model_repo = f"cyberpowder/{model_name}_v{model_version}"

# Create metadata JSON file
metadata_path = os.path.join(temp_dir, f"{model_name}_v{model_version}_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata_props, f, indent=2)

# Create or ensure repository exists
try:
    hf_api.create_repo(model_repo, private=True, token=HF_TOKEN)
    print(f"Created new repository: {model_repo}")
except Exception as e:
    print(f"Repository may already exist or error creating it: {e}")

# Upload ONNX model to Hugging Face
print(f"Uploading ONNX model to Hugging Face: {model_repo}")
hf_api.upload_file(
    path_or_fileobj=temp_model_path,
    repo_id=model_repo,
    path_in_repo=f"{model_name}_v{model_version}.onnx",
    token=HF_TOKEN
)

# Upload metadata to Hugging Face
print(f"Uploading metadata to Hugging Face: {model_repo}")
hf_api.upload_file(
    path_or_fileobj=metadata_path,
    repo_id=model_repo,
    path_in_repo=f"{model_name}_v{model_version}_metadata.json",
    token=HF_TOKEN
)

print(f"Model and metadata successfully uploaded to Hugging Face: {model_repo}")


# %%
# Download and use the model from Hugging Face
from huggingface_hub import hf_hub_download

model_repo = f"cyberpowder/{model_name}_v{model_version}"
model_filename = f"{model_name}_v{model_version}.onnx"
metadata_filename = f"{model_name}_v{model_version}_metadata.json"

# List files in the repo to confirm upload was successful
print(f"Files in repository {model_repo}:")
model_files = hf_api.list_repo_files(model_repo, token=HF_TOKEN)
for file in model_files:
    print(f"  - {file}")

# Create temporary directory for downloaded files
download_dir = tempfile.mkdtemp()

try:
    # Download model from Hugging Face
    print(f"\nDownloading model from Hugging Face...")
    model_path = hf_hub_download(
        repo_id=model_repo,
        filename=model_filename,
        token=HF_TOKEN,
        local_dir=download_dir
    )
    
    # Download metadata
    print(f"Downloading metadata from Hugging Face...")
    metadata_path = hf_hub_download(
        repo_id=model_repo,
        filename=metadata_filename,
        token=HF_TOKEN,
        local_dir=download_dir
    )
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nModel metadata:")
    print(f"  Version: {metadata.get('version')}")
    print(f"  Training date: {metadata.get('training_date')}")
    print(f"  Description: {metadata.get('description')}")
    print(f"  Framework: {metadata.get('framework')}")
    print(f"  Metrics: {metadata.get('metrics')}")
    
    # Load model with ONNX Runtime
    print(f"\nLoading model for inference...")
    session = ort.InferenceSession(model_path)
    
    # Sample data for inference
    sample_inputs = [
        [5.0, 20.0],   # Low CQI, low throughput
        [10.0, 50.0],  # Medium CQI, medium throughput
        [15.0, 100.0], # High CQI, high throughput
        [8.0, 80.0],   # Medium-low CQI, medium-high throughput
        [12.0, 30.0]   # Medium-high CQI, medium-low throughput
    ]
    
    input_tensor = np.array(sample_inputs, dtype=np.float32)
    
    # Run inference
    print(f"\nRunning inference with sample data...")
    outputs = session.run(None, {"input": input_tensor})
    
    # Print results as a table
    print("\nPrediction Results:")
    print("------------------------------------------------------")
    print("   CQI   | Throughput (Mbps) | Predicted min_prb_ratio")
    print("------------------------------------------------------")
    for i, sample in enumerate(sample_inputs):
        print(f"  {sample[0]:5.1f}  |      {sample[1]:7.1f}     |        {outputs[0][i][0]:7.2f}")
    print("------------------------------------------------------")
    
    # Create a visualization
    fig = go.Figure()
    
    # Add points for the samples
    fig.add_trace(go.Scatter3d(
        x=[sample[0] for sample in sample_inputs],  # CQI
        y=[sample[1] for sample in sample_inputs],  # Throughput
        z=[pred[0] for pred in outputs[0]],        # Predicted min_prb_ratio
        mode='markers',
        marker=dict(
            size=8,
            color='red',
        ),
        name='Predictions',
        hovertemplate='CQI: %{x:.1f}<br>Throughput: %{y:.1f} Mbps<br>Predicted PRB: %{z:.1f}%<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Predictions from Hugging Face Model ({model_name} v{model_version})",
        scene=dict(
            xaxis_title='CQI',
            yaxis_title='Throughput (Mbps)',
            zaxis_title='min_prb_ratio (%)',
        ),
        width=800,
        height=600
    )
    
    fig.show()
    
except Exception as e:
    print(f"Error using model from Hugging Face: {e}")
    
finally:
    # Clean up
    import shutil
    shutil.rmtree(download_dir, ignore_errors=True)
