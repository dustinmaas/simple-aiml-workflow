#!/usr/bin/env python3
"""
Script to analyze model performance from inference experiment runner data.

This script:
1. Connects to InfluxDB to retrieve data from student_model_evals bucket
2. Calculates average absolute percentage difference between target_thp and actual_thp 
3. Gets the latest run_id for each model_repo
4. Writes results to a CSV file
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import influxdb_client

# InfluxDB Constants
DEFAULT_INFLUXDB_URL = "http://localhost:8086"
DEFAULT_INFLUXDB_TOKEN = "ric_admin_token"
DEFAULT_INFLUXDB_ORG = "ric"
DEFAULT_INFLUXDB_BUCKET = "student_model_evals"

def get_influxdb_connection():
    """
    Create a connection to InfluxDB using constants.
    
    Returns:
        An InfluxDB client instance
    """
    # Create InfluxDB client using the same approach as experiment_runner.py
    client = influxdb_client.InfluxDBClient(
        url=DEFAULT_INFLUXDB_URL,
        token=DEFAULT_INFLUXDB_TOKEN,
        org=DEFAULT_INFLUXDB_ORG
    )
    
    return client

def query_model_data(query_api):
    """
    Query model evaluation data from InfluxDB.
    
    Args:
        query_api: InfluxDB query API object
        
    Returns:
        pd.DataFrame: DataFrame containing model evaluation data
    """
    # Flux query to get all student model evaluation data
    query = '''
    from(bucket: "student_model_evals")
      |> range(start: 0)
      |> filter(fn: (r) => r._measurement == "student_model_eval")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    # Execute query and convert to DataFrame
    result = query_api.query_data_frame(query)
    
    if result.empty:
        print("No data found in student_model_evals bucket")
        sys.exit(1)
    
    # Make sure we have the necessary columns
    required_columns = ["model_repo", "run_id", "target_thp", "actual_thp"]
    for col in required_columns:
        if col not in result.columns:
            print(f"Required column '{col}' not found in query results")
            sys.exit(1)
    
    return result

def calculate_performance_metrics(df):
    """
    Calculate performance metrics for each model repository.
    
    Args:
        df: DataFrame containing model evaluation data
        
    Returns:
        pd.DataFrame: DataFrame with performance metrics
    """
    # Convert run_id to timestamp (if it's stored as a string timestamp)
    if df['run_id'].dtype == 'object':
        try:
            # Try to convert run_id to int if it's a string timestamp
            df['run_id'] = df['run_id'].astype(int)
        except:
            # If it's not convertible to int, we'll use it as-is
            pass
    
    # Calculate absolute percentage difference
    df['abs_percentage_diff'] = np.abs((df['actual_thp'] - df['target_thp']) / df['target_thp'] * 100)
    
    # Find the latest run_id for each model_repo
    latest_runs = df.groupby('model_repo')['run_id'].max().reset_index()
    latest_runs.rename(columns={'run_id': 'latest_run_id'}, inplace=True)
    
    # Merge to filter only the latest runs
    df = df.merge(latest_runs, on='model_repo')
    df = df[df['run_id'] == df['latest_run_id']]
    
    # Calculate average absolute percentage difference for each model/run
    result = df.groupby(['model_repo', 'latest_run_id'])['abs_percentage_diff'].mean().reset_index()
    result.rename(columns={'abs_percentage_diff': 'avg_absolute_percentage_diff'}, inplace=True)
    
    # Sort by average absolute percentage difference in ascending order
    result.sort_values(by='avg_absolute_percentage_diff', ascending=True, inplace=True)
    
    return result

def main():
    """Main function to analyze model performance and write results to CSV."""
    try:
        # Connect to InfluxDB
        client = get_influxdb_connection()
        query_api = client.query_api()
        
        # Query data from InfluxDB
        print("Querying data from InfluxDB...")
        df = query_model_data(query_api)
        print(f"Retrieved {len(df)} data points")
        
        # Calculate performance metrics
        print("Calculating performance metrics...")
        results = calculate_performance_metrics(df)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"model_performance_{timestamp}.csv"
        
        # Write results to CSV
        results.to_csv(output_file, index=False)
        print(f"Results written to {output_file}")
        
        # Display results
        print("\nResults Summary:")
        print(results.to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    main()
