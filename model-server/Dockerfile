FROM python:3.10-slim

WORKDIR /app

# Install uv using pip
RUN pip install uv

# Install Python dependencies using uv
COPY requirements.txt /app/
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . /app/

# Create directory for model storage
RUN mkdir -p /app/models

# Expose the API port
EXPOSE 5000

# Set the entrypoint
CMD ["python", "app.py"] 