#!/usr/bin/env python3
"""
Simple script to download a model from Hugging Face and test it against provided inputs.

This script:
1. Downloads the model from Hugging Face Hub
2. Tests it against specified CQI and throughput pairs
3. Prints the prediction results
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import huggingface_hub
import onnxruntime as ort

# Add parent directory to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Sample inputs [CQI, Throughput] pairs
SAMPLE_INPUTS = [
    [15, 203419], 
    [14, 161241],
    [13, 122670],
    [12, 112547],
    [11, 119182],
    [10, 92302], 
    [9, 78992],
    [8, 94396],
    [7, 78733],
]

def download_model(repo_id: str, model_file: Optional[str] = None, local_dir: str = "/tmp") -> str:
    """
    Download model from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID
        model_file: Model filename in the repository (if None, will auto-detect)
        local_dir: Directory where the model will be downloaded
        
    Returns:
        Path to the downloaded model
    """
    print(f"Downloading model from {repo_id}...")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Get HuggingFace token if available
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # If model_file not specified, find the first ONNX model
        if not model_file:
            try:
                repo_files = huggingface_hub.list_repo_files(repo_id=repo_id, token=hf_token)
                onnx_files = [f for f in repo_files if f.endswith('.onnx')]
                
                if not onnx_files:
                    raise ValueError(f"No ONNX models found in repository {repo_id}")
                
                model_file = onnx_files[0]
                print(f"Automatically selected ONNX model: {model_file}")
            except Exception as e:
                print(f"Error finding ONNX model in repository: {e}")
                raise
        
        # Download the model file
        model_path = huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=model_file,
            token=hf_token,
            cache_dir=local_dir
        )
        
        print(f"Model downloaded to {model_path}")
        return model_path
    
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

def create_onnx_session(model_path: str) -> ort.InferenceSession:
    """
    Create an ONNX runtime inference session.
    
    Args:
        model_path: Path to the ONNX model
        
    Returns:
        ONNX runtime inference session
    """
    try:
        session_options = ort.SessionOptions()
        providers = ['CPUExecutionProvider']
        
        # Create and return inference session
        return ort.InferenceSession(model_path, providers=providers, sess_options=session_options)
    
    except Exception as e:
        print(f"Error creating ONNX session: {e}")
        sys.exit(1)

def run_prediction(session: ort.InferenceSession, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference with an ONNX model.
    
    Args:
        session: ONNX runtime inference session
        input_data: Input data for the model
        
    Returns:
        Prediction results
    """
    try:
        # Get input and output names
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        
        # Prepare input tensors
        input_tensors = {}
        for name in input_names:
            if name in input_data:
                # Convert input data to numpy array with explicit float32 type
                tensor = np.array(input_data[name], dtype=np.float32)
                
                # Format tensor if needed
                if len(tensor.shape) == 1:
                    tensor = tensor.reshape(1, -1)
                
                input_tensors[name] = tensor
            else:
                raise ValueError(f"Input {name} not found in request data")
        
        # Run inference
        outputs = session.run(output_names, input_tensors)
        
        # Format results
        result = {}
        for i, name in enumerate(output_names):
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(outputs[i], np.ndarray):
                result[name] = outputs[i].tolist()
            else:
                result[name] = outputs[i]
        
        return result
    
    except Exception as e:
        print(f"Error running prediction: {e}")
        sys.exit(1)

def test_model(model_path: str, inputs: List[List[int]]) -> None:
    """
    Test the model with the provided inputs and print results.
    
    Args:
        model_path: Path to the ONNX model
        inputs: List of [CQI, Throughput] pairs to test
    """
    try:
        print("\nRunning model predictions for the given inputs...")
        print("=" * 60)
        print(f"{'CQI':<6} {'Throughput':<12} {'Prediction':<12}")
        print("-" * 60)
        
        # Create ONNX session for the model
        session = create_onnx_session(model_path)
        
        # Process each input pair
        for input_pair in inputs:
            cqi, throughput = input_pair
            
            # Prepare input data
            input_data = {
                "input": [[float(cqi), float(throughput)]]
            }
            
            # Run prediction
            result = run_prediction(session, input_data)
            
            # Get the output values
            if "output" in result:
                prediction = result["output"][0][0]
            else:
                # Try to find the first output key if "output" is not available
                first_output_key = list(result.keys())[0]
                prediction = result[first_output_key][0][0]
            
            # Print the formatted result
            print(f"{cqi:<6} {throughput:<12} {prediction:<12.6f}")
        
        print("=" * 60)
    
    except Exception as e:
        print(f"Error testing model: {e}")
        sys.exit(1)

def main():
    """Main function to parse arguments and run model tests."""
    parser = argparse.ArgumentParser(description='Test model with sample inputs')
    parser.add_argument('--model-repo', required=True, help='HuggingFace repo containing ONNX model')
    parser.add_argument('--model-file', type=str, help='Name of ONNX model file in repo (if not specified, will auto-detect)')
    parser.add_argument('--local-dir', type=str, default='/tmp',
                        help='Local directory to download the model to')
    args = parser.parse_args()
    
    try:
        # Download model from Hugging Face
        model_path = download_model(args.model_repo, args.model_file, args.local_dir)
        
        # Test model with sample inputs
        test_model(model_path, SAMPLE_INPUTS)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
