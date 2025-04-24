#!/usr/bin/env python3
"""
Run inference experiment.

This script runs an inference experiment using a machine learning model
to predict min_prb_ratio values needed to meet throughput targets
at different attenuation levels.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from core.node_controller import NodeController
from core.inference_experiment_runner import InferenceExperimentRunner
from config.inference_config_loader import load_inference_config, load_node_config
from utils.common import setup_logging

logger = logging.getLogger(__name__)

def main():
    """
    Main function for running inference experiments.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run inference experiment')
    parser.add_argument('--config', default='config/inference.yaml', help='Path to experiment config')
    parser.add_argument('--model-repo', required=True, help='Hugging Face repo containing ONNX model')
    parser.add_argument('--model-file', help='Name of ONNX model file in repo (if not specified, will auto-detect)')
    parser.add_argument('--output-dir', default='.', help='Directory for output files')
    parser.add_argument('--experiment-id', help='Experiment ID (defaults to timestamp-based ID)')
    parser.add_argument('--node-config', help='Path to node configuration file')
    parser.add_argument('--log-level', default='INFO', 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    parser.add_argument('--log-file', help='Path to log file (optional)')
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(
        log_level=log_level,
        log_file=args.log_file
    )
    
    # Ensure config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    # Load inference configuration
    try:
        config = load_inference_config(str(config_path))
        logger.info(f"Loaded inference configuration from {args.config}")
    except Exception as e:
        logger.error(f"Error loading inference config file: {e}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for HUGGINGFACE_TOKEN
    if not os.getenv("HUGGINGFACE_TOKEN"):
        logger.warning("HUGGINGFACE_TOKEN environment variable not set - private models may not be accessible")
    
    # Initialize components
    try:
        # Load node configuration (either from command-line argument or default location)
        node_config_path = args.node_config
        if node_config_path is None:
            # Try to locate node_config.yaml relative to the inference config
            relative_path = os.path.join(os.path.dirname(str(config_path)), "node_config.yaml")
            if os.path.exists(relative_path):
                node_config_path = relative_path
        
        node_config = load_node_config(node_config_path)
        logger.info(f"Loaded node configuration from {node_config_path or 'default location'}")
        
        # Initialize node controller
        node_controller = NodeController(node_config)
        
        # Set the output file path
        data_file = output_dir / f"data_{args.experiment_id or 'latest'}.txt"
        
        # Create and run the experiment
        runner = InferenceExperimentRunner(
            node_controller=node_controller,
            config=config,
            model_repo=args.model_repo,
            model_file=args.model_file,
            experiment_id=args.experiment_id,
            data_file=str(data_file)
        )
        
        logger.info(f"Starting experiment with model from {args.model_repo}")
        success = runner.run()
        
        if success:
            logger.info("Experiment completed successfully")
            return 0
        else:
            logger.error("Experiment failed")
            return 1
            
    except Exception as e:
        logger.error(f"Error initializing experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
