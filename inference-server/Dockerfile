FROM python:3.10.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv using pip
RUN pip install uv

# Install Python dependencies using uv
COPY requirements.txt /app/
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . /app/

# Expose the API port
EXPOSE 5000

# Set the entrypoint
CMD ["python", "app.py"]
