FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv using pip
RUN pip install uv

# Install Python dependencies using uv
COPY requirements.txt /app/
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . /app/

# Set execute permissions for scripts
RUN chmod +x /app/run_experiment.py
RUN chmod +x /app/test_torch_inference.py

# Set an environment variable for the model server URL
ENV MODEL_SERVER_URL=http://model-server:5000
ENV INFERENCE_SERVER_URL=http://inference-server:5000

# Run the experiment script when the container starts
CMD ["python", "/app/run_experiment.py"] 