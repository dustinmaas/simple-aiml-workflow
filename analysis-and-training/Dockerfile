FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv using pip
RUN pip install uv

# Install Python dependencies using uv
COPY requirements.txt /app/
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . /app/

# Set up Jupyter configuration
RUN mkdir -p /root/.jupyter
COPY jupyter_notebook_config.py /root/.jupyter/

# Create directories for data and notebooks
RUN mkdir -p /app/data /app/notebooks

# Expose Jupyter port
EXPOSE 8888

# Set the entrypoint
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] 