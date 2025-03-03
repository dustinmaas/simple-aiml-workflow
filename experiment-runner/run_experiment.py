#!/usr/bin/env python3
"""
Main experiment runner script for the AI/ML workflow.
This script serves as the entry point for running experiments in the container.
"""

import os
import sys
import time
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run AI/ML workflow experiments")
    
    # Add command line arguments
    parser.add_argument("--mode", choices=["interactive", "test"], default="interactive",
                        help="Run mode: interactive (default) or test")
    parser.add_argument("--test-model", default="torch_test_model", 
                        help="Model ID to use for testing (applies only in test mode)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("AI/ML Workflow Experiment Runner")
    print("=" * 80)
    
    if args.mode == "test":
        print(f"Running in TEST mode with model ID: {args.test_model}")
        
        # Run the test script
        cmd = [
            "python", 
            "/app/test_torch_inference.py", 
            f"--model-id={args.test_model}"
        ]
        
        # Execute the command and show output
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Test failed with exit code {e.returncode}")
            return e.returncode
            
    else:
        print("Running in INTERACTIVE mode")
        print("The container is now running and waiting for commands.")
        print("You can connect to the container using 'docker exec' to run tests or experiments.")
        print("\nExample commands:")
        print("  - Run the test script: python /app/test_torch_inference.py")
        print("  - Run interactive Python: python")
        
        # Keep the container running
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("Shutting down...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 