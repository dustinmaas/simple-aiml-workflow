#!/usr/bin/env python3
"""
Model Server API for PyTorch models.

This API serves PyTorch models to other services, particularly the inference server.
"""

import os
import sys
import logging
import json
import pickle
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import onnxruntime as ort
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = os.environ.get('MODELS_DIR', '/app/models')

app = Flask(__name__)

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models."""
    try:
        models = []
        for filename in os.listdir(MODELS_DIR):
            if filename.endswith('.pt'):
                model_id = filename.replace('.pt', '')
                models.append({
                    "id": model_id,
                    "file": filename,
                    "path": os.path.join(MODELS_DIR, filename)
                })
        return jsonify(models)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Retrieve a specific model file."""
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pt")
    
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model {model_id} not found"}), 404
    
    try:
        return send_file(model_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error sending model file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models/<model_id>', methods=['POST'])
def upload_model(model_id):
    """Upload and store a model file"""
    if 'model' not in request.files:
        return jsonify({"error": "No model file provided"}), 400
    
    model_file = request.files['model']
    if model_file.filename == '':
        return jsonify({"error": "Empty model filename"}), 400
    
    # Get secure filename and create full path
    filename = secure_filename(model_file.filename)
    filepath = os.path.join(MODELS_DIR, f"{model_id}.pt")
    
    # Save the model file
    model_file.save(filepath)
    
    # Validate model
    try:
        session = load_model(model_id)
        test_input_data = np.array([[10.0, 90.0]], dtype=np.float32)  # Example input (batch size 1, 2 features)

        # Run inference
        outputs = session.run(None, {"input": test_input_data})  # 'input' is the input name used during export
        del session
        logger.info(f"Model {model_id} uploaded successfully")
        logger.info(f"Test inputs/outputs: {test_input_data} -> {outputs}")
        return jsonify({"success": True, "message": f"Model {model_id} uploaded successfully"}), 200
    except Exception as e:
        # If validation fails, remove the file
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Invalid model file: {str(e)}"}), 400

@app.route('/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a model."""
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pt")
    
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model {model_id} not found"}), 404
    
    try:
        os.remove(model_path)
        return jsonify({
            "message": f"Model {model_id} deleted successfully"
        })
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        return jsonify({"error": str(e)}), 500

def load_model(model_id):
    """Load model from disk and return it with metadata"""
    filepath = os.path.join(MODELS_DIR, f"{model_id}.pt")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    try:
        # Load the checkpoint
        model = ort.InferenceSession(filepath)
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 