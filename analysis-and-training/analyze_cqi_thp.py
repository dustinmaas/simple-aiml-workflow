#!/usr/bin/env python3
"""
Script to analyze DRB.UeThpDl statistics per CQI value from InfluxDB data.
Groups DRB.UeThpDl values by CQI values and calculates statistics. Removes
outliers and calculates the 25th, mean, and 75th percentiles.
"""

import os

import pandas as pd
import warnings

from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.warnings import MissingPivotFunction
from tabulate import tabulate


warnings.simplefilter("ignore", MissingPivotFunction)

class Config:
    # InfluxDB connection details (from Docker environment)
    INFLUXDB_HOST = os.environ.get("INFLUXDB_HOST", "datalake_influxdb")
    INFLUXDB_PORT = os.environ.get("INFLUXDB_PORT", "8086")
    INFLUXDB_TOKEN = os.environ.get("INFLUXDB_ADMIN_TOKEN", "ric_admin_token")
    INFLUXDB_ORG = os.environ.get("INFLUXDB_ORG", "ric")
    INFLUXDB_BUCKET = os.environ.get("INFLUXDB_BUCKET", "network_metrics")
    
    # Analysis parameters
    EXPERIMENT_ID = "exp_1745277823"
    UE_ID = "1"
    MIN_CQI = 6
    MAX_CQI = 15

url = f"http://{Config.INFLUXDB_HOST}:{Config.INFLUXDB_PORT}"
client = InfluxDBClient(url=url, token=Config.INFLUXDB_TOKEN, org=Config.INFLUXDB_ORG)
query_api = client.query_api()

# Fetch CQI and DRB.UeThpDl data for UE ID 1 only
print(f"Fetching data from InfluxDB for UE ID {Config.UE_ID}...")
query = f"""
from(bucket: "{Config.INFLUXDB_BUCKET}")
  |> range(start: -30d)
  |> filter(fn: (r) => r.experiment_id == "{Config.EXPERIMENT_ID}")
  |> filter(fn: (r) => r.ue_id == "{Config.UE_ID}")
  |> filter(fn: (r) => r._field == "CQI" or r._field == "DRB.UEThpDl")
  |> pivot(rowKey:["_time", "ue_id", "min_prb_ratio", "max_prb_ratio", "atten"], 
           columnKey: ["_field"], 
           valueColumn: "_value")
  |> filter(fn: (r) => exists r.CQI and exists r["DRB.UEThpDl"])
"""

def remove_outliers(df, column):
    """
    Remove outliers using the IQR method
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

try:
    result = query_api.query_data_frame(query)
    
    if result is None or (isinstance(result, list) and len(result) == 0):
        print("No data found matching the query.")
        client.close()
        exit()
        
    # If result is a list of dataframes, concatenate them
    if isinstance(result, list):
        if len(result) > 0:
            df = pd.concat(result)
        else:
            print("No data found.")
            client.close()
            exit()
    else:
        df = result
    
    if df.empty:
        print("No data found.")
        client.close()
        exit()
        
    # Check if both CQI and DRB.UEThpDl columns exist
    if 'CQI' not in df.columns or 'DRB.UEThpDl' not in df.columns:
        print(f"Required columns not found in data. Available columns: {df.columns.tolist()}")
        client.close()
        exit()
        
    print(f"Retrieved {len(df)} data points with both CQI and DRB.UEThpDl values.")
    
    # Convert CQI to integer for grouping
    df['CQI'] = pd.to_numeric(df['CQI'], errors='coerce')
    df = df.dropna(subset=['CQI'])  # Remove rows with non-numeric CQI
    df['CQI'] = df['CQI'].astype(int)
    
    # Filter for CQI values between 6 and 15
    df = df[(df['CQI'] >= Config.MIN_CQI) & (df['CQI'] <= Config.MAX_CQI)]
    
    # Group by CQI, remove outliers, and calculate statistics
    results = []
    original_counts = []
    cleaned_counts = []
    
    for cqi in range(Config.MIN_CQI, Config.MAX_CQI + 1):
        cqi_data = df[df['CQI'] == cqi]
        original_count = len(cqi_data)
        original_counts.append(original_count)
        
        if not cqi_data.empty:
            # Remove outliers
            cleaned_data = remove_outliers(cqi_data, 'DRB.UEThpDl')
            cleaned_count = len(cleaned_data)
            cleaned_counts.append(cleaned_count)
            
            if not cleaned_data.empty:
                thp_values = cleaned_data['DRB.UEThpDl']
                mean = thp_values.mean()
                percentile_25 = thp_values.quantile(0.25)
                percentile_75 = thp_values.quantile(0.75)
                
                results.append([
                    cqi, 
                    int(round(percentile_25, 0)), 
                    int(round(mean, 0)), 
                    int(round(percentile_75, 0)),
                    cleaned_count,
                    f"{100 * (original_count - cleaned_count) / original_count:.1f}%"
                ])
            else:
                results.append([cqi, "No data", "No data", "No data", 0, "N/A"])
        else:
            results.append([cqi, "No data", "No data", "No data", 0, "N/A"])
            cleaned_counts.append(0)
    
    # Display results
    headers = ["CQI", "25th Percentile", "Mean", "75th Percentile", "Sample Count", "Outliers Removed"]
    print("\nDRB.UeThpDl Statistics by CQI Value (after outlier removal):")
    print(tabulate(results, headers=headers, tablefmt="grid"))
    
    # Verify we're analyzing data for UE ID 1
    if 'ue_id' in df.columns:
        ue_ids = df['ue_id'].unique()
        print(f"\nConfirmed data is for UE ID: {sorted(ue_ids)}")
    
    # Calculate the total number of samples and outliers
    total_original = sum(original_counts)
    total_cleaned = sum(cleaned_counts)
    total_outliers_pct = 100 * (total_original - total_cleaned) / total_original if total_original > 0 else 0
    print(f"\nTotal data points: {total_original}")
    print(f"Total outliers removed: {total_original - total_cleaned} ({total_outliers_pct:.1f}%)")
    
    # Correlations on cleaned data
    cleaned_df = pd.DataFrame()
    for cqi in range(Config.MIN_CQI, Config.MAX_CQI + 1):
        cqi_data = df[df['CQI'] == cqi]
        if not cqi_data.empty:
            cleaned_data = remove_outliers(cqi_data, 'DRB.UEThpDl')
            cleaned_df = pd.concat([cleaned_df, cleaned_data])
    
    if len(cleaned_df) > 5:  # Only calculate correlation if we have enough data points
        correlation = cleaned_df['CQI'].corr(cleaned_df['DRB.UEThpDl'])
        print(f"\nCorrelation between CQI and throughput (after outlier removal): {correlation:.4f}")
    
    # Save results to CSV for inference targets
    os.makedirs("/app/data", exist_ok=True)
    output_csv = f"/app/data/stats_{Config.EXPERIMENT_ID}.csv"
    
    # Create a DataFrame with the rounded statistics and save to CSV
    stats_df = pd.DataFrame([
        {"CQI": row[0], "Throughput_25th": row[1], "Throughput_Mean": row[2], "Throughput_75th": row[3]}
        for row in results if row[2] != "No data"
    ])
    
    stats_df.to_csv(output_csv, index=False)
    print(f"\nSaved throughput statistics to {output_csv}")
    
except Exception as e:
    print(f"Error during query or analysis: {e}")

# Close the client connection
client.close()
