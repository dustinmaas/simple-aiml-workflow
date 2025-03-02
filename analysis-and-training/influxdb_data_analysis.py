#!/usr/bin/env python3
"""
InfluxDB Data Analysis for O-RAN

This script demonstrates how to retrieve data from InfluxDB,
analyze it, and prepare it for model training.

This can be run directly or converted to a Jupyter notebook.
"""

# %% [markdown]
# # InfluxDB Data Analysis for O-RAN
# 
# This notebook demonstrates how to retrieve data from InfluxDB, analyze it, and prepare it for model training.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta

# Configure plotting
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

# %% [markdown]
# ## Connect to InfluxDB
# 
# First, we'll establish a connection to the InfluxDB server where our O-RAN metrics are stored.

# %%
# InfluxDB connection parameters
INFLUXDB_URL = "http://10.0.2.25:8086"  # Use InfluxDB IP or container name
INFLUXDB_TOKEN = "ric_admin_token"
INFLUXDB_ORG = "ric"
INFLUXDB_BUCKET = "network_metrics"

try:
    # Initialize the InfluxDB client
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    print(f"Successfully connected to InfluxDB at {INFLUXDB_URL}")
    
    # Get the query API
    query_api = client.query_api()
except Exception as e:
    print(f"Error connecting to InfluxDB: {e}")
    print("Please ensure InfluxDB is running and accessible.")

# %% [markdown]
# ## Query Recent Data
# 
# Now, let's query some recent data from our database.

# %%
# Define the time range for data retrieval
time_range = "-24h"  # Last 24 hours

# Flux query to get data
flux_query = f'''
from(bucket: "{INFLUXDB_BUCKET}")
  |> range(start: {time_range})
  |> filter(fn: (r) => r._measurement == "network_metrics")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
'''

try:
    # Execute the query and convert to pandas DataFrame
    result = query_api.query_data_frame(flux_query)
    
    if isinstance(result, list):
        if len(result) > 0:
            df = pd.concat(result)
        else:
            print("No data returned from query.")
            df = pd.DataFrame()
    else:
        df = result
        
    if not df.empty:
        # Display basic information about the data
        print(f"Retrieved {len(df)} rows of data")
        print(f"Columns: {df.columns.tolist()}")
        print(df.head())
    else:
        print("No data available for the specified time range.")
except Exception as e:
    print(f"Error querying data: {e}")
    print("Creating synthetic data for demonstration")
    
    # Synthetic data generation
    num_samples = 1000
    time_index = pd.date_range(end=datetime.now(), periods=num_samples, freq='10T')
    
    df = pd.DataFrame({
        '_time': time_index,
        'ue_id': np.random.choice([1, 3], size=num_samples),
        'RSRP': np.random.uniform(-120, -80, num_samples),
        'RSRQ': np.random.uniform(-20, -5, num_samples),
        'CQI': np.random.randint(1, 16, num_samples),
        'min_prb_ratio': np.random.uniform(0, 100, num_samples),
        'DRB.UEThpDl': np.random.uniform(1, 500, num_samples),  # Downlink throughput
        'DRB.UEThpUl': np.random.uniform(1, 100, num_samples)   # Uplink throughput
    })
    
    print(df.head())

# %% [markdown]
# ## Data Visualization
# 
# Let's visualize some of the key metrics to understand patterns and relationships.

# %%
if not df.empty and 'RSRP' in df.columns and 'DRB.UEThpDl' in df.columns:
    # Make sure time column is datetime
    if '_time' in df.columns:
        df['_time'] = pd.to_datetime(df['_time'])
    
    # Plot RSRP over time
    plt.figure(figsize=(14, 6))
    
    if 'ue_id' in df.columns:
        for ue_id in df['ue_id'].unique():
            subset = df[df['ue_id'] == ue_id]
            plt.plot(subset['_time'], subset['RSRP'], label=f'UE ID: {ue_id}')
    else:
        plt.plot(df['_time'], df['RSRP'])
        
    plt.title('RSRP Over Time')
    plt.ylabel('RSRP (dBm)')
    plt.xlabel('Time')
    plt.grid(True)
    plt.legend()
    plt.savefig('rsrp_over_time.png')
    plt.close()
    
    # Scatter plot to show relationship between RSRP and throughput
    plt.figure(figsize=(10, 8))
    
    if 'ue_id' in df.columns:
        for ue_id in df['ue_id'].unique():
            subset = df[df['ue_id'] == ue_id]
            plt.scatter(subset['RSRP'], subset['DRB.UEThpDl'], alpha=0.6, label=f'UE ID: {ue_id}')
    else:
        plt.scatter(df['RSRP'], df['DRB.UEThpDl'], alpha=0.6)
    
    plt.title('Relationship Between RSRP and Downlink Throughput')
    plt.xlabel('RSRP (dBm)')
    plt.ylabel('Downlink Throughput (Mbps)')
    plt.grid(True)
    plt.legend()
    plt.savefig('rsrp_vs_throughput.png')
    plt.close()

# %% [markdown]
# ## Analysis by PRB Ratio
# 
# Let's analyze how the min_prb_ratio affects performance.

# %%
if not df.empty and 'min_prb_ratio' in df.columns and 'DRB.UEThpDl' in df.columns:
    # Group data by min_prb_ratio ranges
    df['prb_range'] = pd.cut(df['min_prb_ratio'], bins=5)
    
    # Calculate average throughput for each PRB range
    prb_performance = df.groupby('prb_range')['DRB.UEThpDl'].agg(['mean', 'std', 'count']).reset_index()
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.bar(prb_performance['prb_range'].astype(str), prb_performance['mean'], yerr=prb_performance['std'])
    plt.title('Downlink Throughput vs PRB Ratio')
    plt.xlabel('PRB Ratio Range')
    plt.ylabel('Average Throughput (Mbps)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.savefig('throughput_vs_prb.png')
    plt.close()
    
    # Show the numerical results
    print(prb_performance)

# %% [markdown]
# ## Correlation Analysis
# 
# Let's look at correlations between different metrics.

# %%
if not df.empty:
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        # Create correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.savefig('correlation_matrix.png')
        plt.close()
        
        print(corr_matrix)

# %% [markdown]
# ## Export Data for Model Training
# 
# Let's prepare and export the data for model training.

# %%
if not df.empty:
    # Select relevant features for modeling
    features = ['RSRP', 'RSRQ', 'CQI', 'min_prb_ratio']
    target = 'DRB.UEThpDl'
    
    # Check if all required columns exist
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
    else:
        # Prepare dataset
        model_data = df[features + [target]].copy()
        model_data = model_data.dropna()
        
        print(f"Prepared dataset with {len(model_data)} rows")
        print(model_data.head())
        
        # Export to CSV
        try:
            model_data.to_csv('/app/data/training_data.csv', index=False)
            print("Data exported to /app/data/training_data.csv")
        except Exception as e:
            print(f"Error saving data: {e}")
            print("Saving to local file instead")
            model_data.to_csv('training_data.csv', index=False)
            print("Data saved to training_data.csv in current directory")

# %% [markdown]
# ## Next Steps
# 
# With this data, you can now:
# 
# 1. Train a model to predict throughput based on radio conditions and PRB ratio
# 2. Optimize PRB allocation to maximize performance
# 3. Create a real-time monitoring dashboard for network performance

# %%
print("Analysis completed successfully!")

if __name__ == "__main__":
    print("To convert this script to a Jupyter notebook, run:")
    print("jupyter nbconvert --to notebook --execute influxdb_data_analysis.py") 