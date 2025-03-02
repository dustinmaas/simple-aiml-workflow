#!/bin/bash
# Convert Python scripts to Jupyter notebooks

# Create notebooks directory if it doesn't exist
mkdir -p /app/notebooks

# Convert influxdb_data_analysis.py to notebook
jupyter nbconvert --to notebook --execute --output-dir=/app/notebooks /app/influxdb_data_analysis.py

echo "Conversion complete - notebooks available in /app/notebooks" 