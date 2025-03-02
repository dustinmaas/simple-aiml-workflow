FROM python:3.10-slim

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

# Make the script executable
RUN chmod +x runner.py

# Set the entrypoint
ENTRYPOINT ["python", "runner.py"]
CMD ["--config=default"] 