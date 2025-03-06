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
# import onnx
# import onnxruntime

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
        self.register_buffer('x_mean', torch.zeros(2))
        self.register_buffer('x_std', torch.ones(2))
        self.register_buffer('y_mean', torch.zeros(1))
        self.register_buffer('y_std', torch.ones(1))

    def forward(self, x: torch.Tensor, denormalize: bool = False) -> torch.Tensor:
        x_scaled = (x - self.x_mean) / self.x_std
        output = self.linear(x_scaled)
        if denormalize:  # Explicit comparison for TorchScript compatibility
            output = output * self.y_std + self.y_mean
        return output
    
    # Add methods to make TorchScript serializable
    def __getstate__(self):
        return {
            'linear.weight': self.linear.weight,
            'linear.bias': self.linear.bias,
            'x_mean': self.x_mean,
            'x_std': self.x_std,
            'y_mean': self.y_mean,
            'y_std': self.y_std
        }
    
    def __setstate__(self, state):
        self.__init__()
        self.linear.weight.data.copy_(state['linear.weight'])
        self.linear.bias.data.copy_(state['linear.bias'])
        self.x_mean.copy_(state['x_mean'])
        self.x_std.copy_(state['x_std'])
        self.y_mean.copy_(state['y_mean'])
        self.y_std.copy_(state['y_std'])
        
    # Add a scripting method to convert the model to TorchScript
    def to_torchscript(self):
        self.eval()  # Set to evaluation mode
        return torch.jit.script(self)


# %%
# Create and train the model
model = LinearRegressionModel()
model.x_mean = X.mean(dim=0, keepdim=True)
model.x_std = X.std(dim=0, keepdim=True)
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

    if (epoch) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# %%
# look at the hyperplane fit

# Get the learned parameters
learned_weights = model.linear.weight.data.cpu().numpy()
learned_bias = model.linear.bias.data.cpu().numpy()

# Print the learned hyperplane equation (in scaled space)
print("\nLearned Hyperplane (in scaled space):")
print(f"y_scaled = {learned_weights[0][0]:.2f}*x1_scaled + {learned_weights[0][1]:.2f}*x2_scaled + {learned_bias[0]:.2f}")

# Create scaled versions of features and targets using the model's normalization parameters
features_scaled = (X.cpu().numpy() - model.x_mean.cpu().numpy()) / model.x_std.cpu().numpy()
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

# Convert meshgrid to scaled space for prediction
X1_scaled = (X1 - model.x_mean.cpu().numpy()[0, 0]) / model.x_std.cpu().numpy()[0, 0]
X2_scaled = (X2 - model.x_mean.cpu().numpy()[0, 1]) / model.x_std.cpu().numpy()[0, 1]

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
model_id = "linear_regression_v1"
temp_dir = tempfile.mkdtemp()
temp_model_path = os.path.join(temp_dir, f"{model_id}.pt")

# Set model to evaluation mode
model.eval()

# Convert the model to TorchScript
scripted_model = model.to_torchscript()

# Save model
scripted_model.save(temp_model_path)

# upload to huggingface
model_name = "linear_regression_v1"
model_repo = f"cyberpowder/{model_name}"
hf_api.upload_file(
    repo_id=model_repo,
    path=temp_model_path,
    token=HF_TOKEN
)

# # Send to model server via REST API
# with open(temp_model_path, 'rb') as f:
#     files = {'model': f}
#     response = requests.post(f'http://model-server:5000/models/{model_id}', files=files)
    
# if response.status_code == 200:
#     print("Model successfully uploaded to model server")
# else:
#     print(f"Error uploading model: {response.text}")
    
# # Cleanup temp file
# os.remove(temp_model_path)
# os.rmdir(temp_dir)



# %%

