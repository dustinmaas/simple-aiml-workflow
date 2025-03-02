#!/usr/bin/env python3
"""
Model Server API for PyTorch models.

This API serves PyTorch models to other services, particularly the inference server.
"""

import os
import logging
from flask import Flask, request, jsonify, send_file
import torch

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
    """Upload a new PyTorch model."""
    try:
        if 'model' not in request.files:
            return jsonify({"error": "No model file provided"}), 400
        
        model_file = request.files['model']
        model_path = os.path.join(MODELS_DIR, f"{model_id}.pt")
        
        # Save the uploaded file
        model_file.save(model_path)
        
        # Verify it's a valid PyTorch model
        try:
            torch.load(model_path)
        except Exception as e:
            # If not a valid model, delete the file and return error
            os.remove(model_path)
            return jsonify({"error": f"Invalid PyTorch model: {str(e)}"}), 400
        
        return jsonify({
            "message": f"Model {model_id} uploaded successfully",
            "id": model_id,
            "path": model_path
        })
    
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        return jsonify({"error": str(e)}), 500

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 