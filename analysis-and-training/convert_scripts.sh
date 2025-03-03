#!/bin/bash
# Convert Python scripts to Jupyter notebooks

NOTEBOOKS_DIR="/app/notebooks"

# Create notebooks directory if it doesn't exist
mkdir -p "$NOTEBOOKS_DIR"

# Convert Python scripts to notebooks using jupytext
for nb in $(ls /app/notebooks/*.py); do
    echo "Converting $nb to ${nb%.py}.ipynb"
    jupytext --to notebook "$nb" --output "${nb%.py}.ipynb"
done
