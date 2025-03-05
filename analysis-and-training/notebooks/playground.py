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
        self.register_buffer('x_mean', torch.zeros(2))
        self.register_buffer('x_std', torch.ones(2))
        self.register_buffer('y_mean', torch.zeros(1))
        self.register_buffer('y_std', torch.ones(1))

    def forward(self, x: torch.Tensor, denormalize: bool = False) -> torch.Tensor:
        x_scaled = (x - self.x_mean) / self.x_std
        output = self.linear(x_scaled)
        if denormalize == True:  # Explicit comparison for TorchScript compatibility
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

# Send to model server via REST API
with open(temp_model_path, 'rb') as f:
    files = {'model': f}
    response = requests.post(f'http://model-server:5000/models/{model_id}', files=files)
    
if response.status_code == 200:
    print("Model successfully uploaded to model server")
else:
    print(f"Error uploading model: {response.text}")
    
# Cleanup temp file
os.remove(temp_model_path)
os.rmdir(temp_dir)

# %%
# Get model from model server
model_id = "linear_regression_v1"
response = requests.get(f'http://model-server:5000/models/{model_id}')

if response.status_code == 200:
    # Create temp file to save the downloaded model
    temp_dir = tempfile.mkdtemp()
    temp_model_path = os.path.join(temp_dir, f"{model_id}.pt")
    
    # Save the model to the temp file
    with open(temp_model_path, 'wb') as f:
        f.write(response.content)
    
    # Load the checkpoint with the model
    loaded_model = torch.jit.load(temp_model_path)
    
    # Verify model parameters
    print(f"Model ID: {model_id}")
    print(f"X mean: {loaded_model.x_mean}")
    print(f"X std: {loaded_model.x_std}")
    print(f"Y mean: {loaded_model.y_mean}")
    print(f"Y std: {loaded_model.y_std}")
    
    # Cleanup
    os.remove(temp_model_path)
    os.rmdir(temp_dir)
    
    print("Model successfully loaded from model server")
else:
    print(f"Error downloading model: {response.text}")




# %%
# generate some predictions across a range of input vals
with torch.no_grad():
    test_input = np.zeros((10 * 16, 2))
    for ii in range(10):
        test_input[ii * 16:ii * 16 + 16, 0] = ii + 6
        test_input[ii * 16:ii * 16 + 16, 1] = np.arange(50, 210, 10)
    test_input = torch.tensor(test_input, dtype=torch.float).to(device)
    
    # Make prediction with denormalization
    predicted = loaded_model(test_input, denormalize=True)
    
    print(f'Predicted values:')
    print(np.concatenate((test_input.cpu().numpy(), predicted.cpu().numpy()), axis=1))


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

