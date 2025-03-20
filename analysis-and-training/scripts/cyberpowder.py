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

# %% [markdown]
"""
# Dataset Analysis and Model Creation/Training


1. Brief HuggingFace intro
1. Creating and uploading datasets to HuggingFace
1. Downloading datasets from HuggingFace
1. Analyzing and visualizing the dataset
1. Linear Regression Model Definition and Training
1. Linear Regression Hyperplane Fit
1. (leave as exercise, then go over) Polynomial Regression Model Definition and Training
1. Comparing Linear and Polynomial Regression Models
1. Uploading the Polynomial Regression Model to Hugging Face
1. Testing the Uploaded Polynomial Model

### Prerequisites

- Join the HuggingFace CyberPowder organization: [link](https://huggingface.co/organizations/cyberpowder/share/VkAxpCJJIebrTqXgdMFxRElyHhnyAJocHQ)
- Review Section VI (AI/ML Workflows) of the [NEU ORAN paper](https://utah.instructure.com/courses/1045795/files/170447527?wrap=1) 
- Read Chapter 2 (Supervised Learning) of [Understanding Deep Learning by Simon Prince](https://github.com/udlbook/udlbook/releases/download/v5.00/UnderstandingDeepLearning_11_21_24_C.pdf)
- maybe split into separate notebooks for each section; reveal each section as we go through it

"""

# %%
# Imports and setup
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
import datetime
import json
import shutil
from huggingface_hub import hf_hub_download

# Get Hugging Face token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("Warning: HF_TOKEN environment variable not found. You may need to set it for Hugging Face operations.")

# %% [markdown]
# ## 1. Creating and Uploading Datasets to Hugging Face

# %%
# Load the original dataset
df = pd.read_csv('/app/data/data.csv')

# %%
# Convert the pandas DataFrame to a Hugging Face Dataset
dataset = datasets.Dataset.from_pandas(df)

# Display basic information about the dataset
print(f"Dataset has {len(dataset)} rows and the following features:")
print(dataset.features)

# %%
# Set the repository name for your dataset
repo_name = "cyberpowder/cyberpowder-network-metrics"

# Create the repository if it doesn't exist
hf_api = hf.HfApi()
try:
    hf_api.create_repo(repo_name, private=True, token=HF_TOKEN)
    print(f"Created new repository: {repo_name}")
except Exception as e:
    print(f"Repository may already exist or error creating it: {e}")

# %%
# Push the dataset to the Hugging Face Hub
dataset.push_to_hub(
    repo_name,
    private=True,  # Set to False to make it publicly accessible
    token=HF_TOKEN,
)

print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_name}")

# %%
# Create a filtered dataset for UE1
df = dataset.to_pandas()

# Filter for ue_id 1 only
print(f"Original dataset size: {len(df)} rows")
df = df[df['ue_id'] == 1]
print(f"After filtering for ue_id 1: {len(df)} rows")

# Create a new dataset from the processed DataFrame
ml_dataset = datasets.Dataset.from_pandas(df)

# %%
# Push the UE1 dataset with the ML-ready configuration
ml_dataset.push_to_hub(
    repo_name,
    config_name="ue1_ml_ready",  # Creates a new configuration in the same repo
    private=True,
    token=HF_TOKEN,
)

print(f"UE1 ML-ready dataset configuration successfully pushed to {repo_name}")
print(f"To load this specific configuration: datasets.load_dataset('{repo_name}', 'ue1_ml_ready')")

# %% [markdown]
# ## 2. Downloading Datasets from Hugging Face

# %%
# Load the dataset from Hugging Face
dataset_ue1 = datasets.load_dataset(repo_name, 'ue1_ml_ready', token=HF_TOKEN)
print(f"Successfully loaded dataset with {len(dataset_ue1['train'])} samples")

# %%
# Convert to pandas and prepare for analysis
df = dataset_ue1['train'].to_pandas()
df['timestamp'] = pd.to_datetime(df['timestamp'])
print("Dataset statistics:")
df.describe()

# %% [markdown]
# ## 3. Visualizing the UE1 Dataset

# %%
# Time series visualization of key metrics
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
# Throughput vs. CQI by min_prb_ratio
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
# Prepare dataset for model training
device = 'cpu'

dataset_ue1.set_format(type='torch', columns=['CQI', 'DRB.UEThpDl', 'min_prb_ratio'], dtype=torch.float32)

# Convert features and targets to PyTorch tensors
X = torch.stack([dataset_ue1['train']['CQI'], dataset_ue1['train']['DRB.UEThpDl']], dim=1)
y = dataset_ue1['train']['min_prb_ratio'].unsqueeze(-1)

# %% [markdown]
# ## 4. Linear Regression Model Definition and Training

# %%
# Define the linear regression model
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
# Create and train the linear model
model = LinearRegressionModel()
model.y_mean = y.mean(dim=0, keepdim=True)
model.y_std = y.std(dim=0, keepdim=True)
model.to(device)
X.to(device)
y.to(device)
criterion = torch.nn.MSELoss() # Mean Squared Error
optimizer = torch.optim.SGD(model.parameters(), lr=.05)

# Train the model
num_epochs = 500
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

# %% [markdown]
# ## 5. Visualizing the Linear Regression Hyperplane Fit

# %%
# Examine the hyperplane fit
model.eval()

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
    title='Linear Regression Hyperplane Fit (Unscaled Values)',
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

fig.show()

# %% [markdown]
# ## 6. Polynomial Regression Model Definition and Training

# %%
# Define the polynomial regression model
class PolynomialRegressionModel(torch.nn.Module):
    def __init__(self, degree=2):
        super(PolynomialRegressionModel, self).__init__()
        self.degree = degree
        
        # Calculate number of polynomial features for 2 input features with degree n
        # For 2 features with degree 2: x1, x2, x1^2, x1*x2, x2^2 = 5 features
        n_poly_features = int((degree + 1) * (degree + 2) / 2) - 1  # -1 because we start from degree 1, not 0
        
        # Apply batch normalization to expanded polynomial features
        self.batch_norm = torch.nn.BatchNorm1d(n_poly_features)
        
        # Linear layer now accepts polynomial features as input
        self.linear = torch.nn.Linear(n_poly_features, 1)
        
        # Register buffers to store the mean and standard deviation of the output features
        self.register_buffer('y_mean', torch.zeros(1))
        self.register_buffer('y_std', torch.ones(1))

    def _polynomial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate polynomial features up to the specified degree.
        For input [x1, x2], with degree=2, this generates [x1, x2, x1^2, x1*x2, x2^2]
        """
        batch_size = x.shape[0]
        x1 = x[:, 0].view(-1, 1)
        x2 = x[:, 1].view(-1, 1)
        
        # Start with degree 1 terms (original features)
        poly_features = [x1, x2]
        
        # Add higher degree terms
        for d in range(2, self.degree + 1):
            for i in range(d + 1):
                # Add term x1^(d-i) * x2^i
                term = torch.pow(x1, d-i) * torch.pow(x2, i)
                poly_features.append(term)
        
        # Concatenate all features
        return torch.cat(poly_features, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First transform input to polynomial features
        x_poly = self._polynomial_features(x)
        
        # Then apply batch normalization to polynomial features
        x_poly_normalized = self.batch_norm(x_poly)
        
        # Apply linear transformation to normalized polynomial features
        output = self.linear(x_poly_normalized)
        
        # Denormalize output during inference
        if not self.training:
            with torch.no_grad():
                output = output * self.y_std + self.y_mean
                
        return output

# %%
# Create and train the polynomial regression model
poly_model = PolynomialRegressionModel(degree=2)  # Using degree 2 polynomial
poly_model.y_mean = y.mean(dim=0, keepdim=True)
poly_model.y_std = y.std(dim=0, keepdim=True)
poly_model.to(device)

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(poly_model.parameters(), lr=0.01)

# Train the model
num_epochs = 10000
for epoch in range(num_epochs):
    poly_model.train()
    # Forward pass
    y_predicted = poly_model(X)
    loss = criterion(y_predicted, (y - poly_model.y_mean) / poly_model.y_std)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# %% [markdown]
# ## 7. Comparing Linear and Polynomial Regression Models

# %%
# Set both models to evaluation mode for fair comparison
model.eval()
poly_model.eval()

# Calculate comprehensive performance metrics
with torch.no_grad():
    # Get raw predictions from both models
    linear_preds = model(X)
    poly_preds = poly_model(X)
    
    # Calculate MSE directly against raw targets
    linear_mse = torch.nn.functional.mse_loss(linear_preds, y).item()
    poly_mse = torch.nn.functional.mse_loss(poly_preds, y).item()
    
    # Calculate R² score (coefficient of determination)
    y_mean = torch.mean(y)
    total_variance = torch.sum((y - y_mean)**2)
    linear_residual_variance = torch.sum((y - linear_preds)**2)
    poly_residual_variance = torch.sum((y - poly_preds)**2)
    
    linear_r2 = (1 - linear_residual_variance / total_variance).item()
    poly_r2 = (1 - poly_residual_variance / total_variance).item()
    
    # Calculate mean absolute error
    linear_mae = torch.mean(torch.abs(linear_preds - y)).item()
    poly_mae = torch.mean(torch.abs(poly_preds - y)).item()

# Print comparison results
print(f"Performance Metrics Comparison:")
print(f"{'Metric':<20} {'Linear':<15} {'Polynomial':<15} {'Improvement':<15}")
print(f"{'-'*60}")
print(f"{'MSE':<20} {linear_mse:<15.6f} {poly_mse:<15.6f} {(1 - poly_mse/linear_mse)*100:<15.2f}%")
print(f"{'MAE':<20} {linear_mae:<15.6f} {poly_mae:<15.6f} {(1 - poly_mae/linear_mae)*100:<15.2f}%")
print(f"{'R² Score':<20} {linear_r2:<15.6f} {poly_r2:<15.6f} {(poly_r2 - linear_r2)*100:<15.2f}%")

# %%
# Visual comparison of both models
model.eval()
poly_model.eval()

# Create a grid of points for visualization
grid_size = 30
x1_range = np.linspace(features_unscaled[:,0].min(), features_unscaled[:,0].max(), grid_size)
x2_range = np.linspace(features_unscaled[:,1].min(), features_unscaled[:,1].max(), grid_size)
X1, X2 = np.meshgrid(x1_range, x2_range)

# Flatten the grid points for prediction
grid_points = np.column_stack([X1.flatten(), X2.flatten()])
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

# Get predictions from both models
with torch.no_grad():
    poly_predictions = poly_model(grid_tensor).cpu().numpy().reshape(grid_size, grid_size)
    linear_predictions = model(grid_tensor).cpu().numpy().reshape(grid_size, grid_size)

# Create 3D visualization comparing both models
fig = make_subplots(
    rows=1, 
    cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    subplot_titles=["Linear Regression Surface", "Polynomial Regression Surface"]
)

# Sample a subset of data points for visualization
sample_indices = np.random.choice(len(features_unscaled), size=min(300, len(features_unscaled)), replace=False)
sample_features = features_unscaled[sample_indices]
sample_targets = targets_unscaled[sample_indices]

# Add data points to both subplots
for i in range(1, 3):
    fig.add_trace(
        go.Scatter3d(
            x=sample_features[:,0],
            y=sample_features[:,1],
            z=sample_targets.flatten(),
            mode='markers',
            marker=dict(
                size=3,
                color='red',
                opacity=0.5
            ),
            name='Data Points',
            showlegend=False,
            hovertemplate='CQI: %{x:.2f}<br>Throughput: %{y:.2f} Mbps<br>min_prb_ratio: %{z:.2f}<extra></extra>'
        ),
        row=1, col=i
    )

# Add linear regression surface
fig.add_trace(
    go.Surface(
        x=X1, 
        y=X2, 
        z=linear_predictions,
        colorscale='Blues',
        opacity=0.7,
        showscale=False,
        name='Linear Regression'
    ),
    row=1, col=1
)

# Add polynomial regression surface
fig.add_trace(
    go.Surface(
        x=X1, 
        y=X2, 
        z=poly_predictions,
        colorscale='Greens',
        opacity=0.7,
        showscale=False,
        name='Polynomial Regression'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title='Comparison of Linear vs. Polynomial Regression Models',
    height=600,
    width=1200,
)

# Update scene settings for both subplots
for i in range(1, 3):
    fig.update_scenes(
        xaxis_title='CQI',
        yaxis_title='Throughput (Mbps)',
        zaxis_title='min_prb_ratio',
        aspectmode='auto',
        row=1, col=i
    )

fig.show()

# %% [markdown]
# ## 8. Uploading the Polynomial Regression Model to Hugging Face

# %%
# Save the polynomial regression model to ONNX and upload to Hugging Face
poly_model_name = "polynomial_regression_model"
poly_model_version = "1.0.0"  # First version

# Calculate comprehensive metrics for metadata
with torch.no_grad():
    poly_preds = poly_model(X)
    poly_mse = torch.nn.functional.mse_loss(poly_preds, y).item()
    poly_mae = torch.mean(torch.abs(poly_preds - y)).item()
    
    # Calculate R² score (coefficient of determination)
    y_mean = torch.mean(y)
    total_variance = torch.sum((y - y_mean)**2)
    poly_residual_variance = torch.sum((y - poly_preds)**2)
    poly_r2 = (1 - poly_residual_variance / total_variance).item()

# Metadata with polynomial degree information and comprehensive metrics
poly_metadata_props = {
    "version": poly_model_version,
    "training_date": datetime.datetime.now().isoformat(),
    "framework": f"PyTorch {torch.__version__}",
    "dataset": "network_metrics_exp_1741030459",
    "metrics": json.dumps({
        "mse": poly_mse,
        "mae": poly_mae,
        "r2": poly_r2
    }),
    "description": f"Polynomial regression model (degree {poly_model.degree}) for PRB prediction based on CQI and throughput",
    "input_features": json.dumps(["CQI", "DRB.UEThpDl"]),
    "output_features": json.dumps(["min_prb_ratio"]),
    "polynomial_degree": poly_model.degree,
    "model_type": "polynomial_regression"
}

# Create temp directory
poly_temp_dir = tempfile.mkdtemp()
poly_model_path = os.path.join(poly_temp_dir, f"{poly_model_name}_v{poly_model_version}.onnx")

# Export the model to ONNX
dummy_input = torch.randn(1, 2)  # Example input
torch.onnx.export(
    poly_model, 
    dummy_input, 
    poly_model_path, 
    verbose=True, 
    input_names=["input"], 
    output_names=["output"], 
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

# Create repository for polynomial model
poly_model_repo = f"cyberpowder/{poly_model_name}_v{poly_model_version}"

# Create metadata JSON file
poly_metadata_path = os.path.join(poly_temp_dir, f"{poly_model_name}_v{poly_model_version}_metadata.json")
with open(poly_metadata_path, 'w') as f:
    json.dump(poly_metadata_props, f, indent=2)

# Create or ensure repository exists
try:
    hf_api.create_repo(poly_model_repo, private=True, token=HF_TOKEN)
    print(f"Created new repository: {poly_model_repo}")
except Exception as e:
    print(f"Repository may already exist or error creating it: {e}")

# Upload ONNX model to Hugging Face
print(f"Uploading polynomial ONNX model to Hugging Face: {poly_model_repo}")
hf_api.upload_file(
    path_or_fileobj=poly_model_path,
    repo_id=poly_model_repo,
    path_in_repo=f"{poly_model_name}_v{poly_model_version}.onnx",
    token=HF_TOKEN
)

# Upload metadata to Hugging Face
print(f"Uploading polynomial model metadata to Hugging Face: {poly_model_repo}")
hf_api.upload_file(
    path_or_fileobj=poly_metadata_path,
    repo_id=poly_model_repo,
    path_in_repo=f"{poly_model_name}_v{poly_model_version}_metadata.json",
    token=HF_TOKEN
)

print(f"Polynomial model and metadata successfully uploaded to Hugging Face: {poly_model_repo}")

# %% [markdown]
# ## Testing the Uploaded Polynomial Model

# %%
#TODO need to fix this to only use possible ranges.
# Download and test the uploaded polynomial model
poly_model_repo = f"cyberpowder/{poly_model_name}_v{poly_model_version}"
poly_model_filename = f"{poly_model_name}_v{poly_model_version}.onnx"
poly_metadata_filename = f"{poly_model_name}_v{poly_model_version}_metadata.json"

# List files in the repo to confirm upload was successful
print(f"Files in repository {poly_model_repo}:")
model_files = hf_api.list_repo_files(poly_model_repo, token=HF_TOKEN)
for file in model_files:
    print(f"  - {file}")

# Create temporary directory for downloaded files
poly_download_dir = tempfile.mkdtemp()

try:
    # Download model from Hugging Face
    print(f"\nDownloading polynomial model from Hugging Face...")
    poly_model_path = hf_hub_download(
        repo_id=poly_model_repo,
        filename=poly_model_filename,
        token=HF_TOKEN,
        local_dir=poly_download_dir
    )
    
    # Download metadata
    print(f"Downloading polynomial model metadata from Hugging Face...")
    poly_metadata_path = hf_hub_download(
        repo_id=poly_model_repo,
        filename=poly_metadata_filename,
        token=HF_TOKEN,
        local_dir=poly_download_dir
    )
    
    # Load metadata
    with open(poly_metadata_path, 'r') as f:
        poly_metadata = json.load(f)
    
    print(f"\nPolynomial model metadata:")
    print(f"  Version: {poly_metadata.get('version')}")
    print(f"  Polynomial degree: {poly_metadata.get('polynomial_degree')}")
    print(f"  Description: {poly_metadata.get('description')}")
    print(f"  Framework: {poly_metadata.get('framework')}")
    print(f"  Metrics: {poly_metadata.get('metrics')}")
    
    # Load model with ONNX Runtime
    print(f"\nLoading polynomial model for inference...")
    poly_session = ort.InferenceSession(poly_model_path)
    
    # Sample data for inference
    sample_inputs = [
        [5.0, 20.0],   # Low CQI, low throughput
        [10.0, 50.0],  # Medium CQI, medium throughput
        [15.0, 100.0], # High CQI, high throughput
        [8.0, 80.0],   # Medium-low CQI, medium-high throughput
        [12.0, 30.0]   # Medium-high CQI, medium-low throughput
    ]
    
    input_tensor = np.array(sample_inputs, dtype=np.float32)
    
    # Run inference with polynomial model
    print(f"\nRunning inference with sample data...")
    poly_outputs = poly_session.run(None, {"input": input_tensor})
    
    # Print results as a table
    print("\nPrediction Results:")
    print("------------------------------------------------------")
    print("   CQI   | Throughput (Mbps) | Predicted min_prb_ratio")
    print("------------------------------------------------------")
    for i, sample in enumerate(sample_inputs):
        print(f"  {sample[0]:5.1f}  |      {sample[1]:7.1f}     |        {poly_outputs[0][i][0]:7.2f}")
    print("------------------------------------------------------")
    
    # Create a visualization
    fig = go.Figure()
    
    # Add points for the samples
    fig.add_trace(go.Scatter3d(
        x=[sample[0] for sample in sample_inputs],  # CQI
        y=[sample[1] for sample in sample_inputs],  # Throughput
        z=[pred[0] for pred in poly_outputs[0]],    # Predicted min_prb_ratio
        mode='markers',
        marker=dict(
            size=8,
            color='green',
            symbol='diamond'
        ),
        name='Polynomial Model Predictions',
        hovertemplate='CQI: %{x:.1f}<br>Throughput: %{y:.1f} Mbps<br>Predicted PRB: %{z:.1f}%<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Predictions from Polynomial Regression Model (Degree {poly_metadata.get('polynomial_degree')})",
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
    print(f"Error using polynomial model from Hugging Face: {e}")
    
finally:
    # Clean up
    shutil.rmtree(poly_download_dir, ignore_errors=True)
    print("Test completed and temporary files cleaned up")

# %%
