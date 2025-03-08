#!/usr/bin/env python3
"""
Main entry point for experiment runner.

This module initializes and runs the experiment runner with the specified configuration.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional, List

from config.config_loader import (
    load_experiment_config,
    load_node_config
)
from core.node_controller import NodeController
from core.experiment_runner import ExperimentRunner
from utils.common import setup_logging

logger: logging.Logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Experiment runner for network measurements"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "test"],
        help="Experiment configuration to use"
    )
    
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="Experiment ID (defaults to timestamp-based ID)"
    )
    
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID (defaults to UUID-based ID)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="10.0.2.1",
        help="API server host"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=61611,
        help="API server port"
    )
    
    parser.add_argument(
        "--data-file",
        type=str,
        default="data.txt",
        help="Path to data output file"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (optional)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    return parser.parse_args()

def main() -> int:
    """
    Main entry point for the experiment runner.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse command line arguments
    args: argparse.Namespace = parse_arguments()
    
    # Set up logging
    log_level: int = getattr(logging, args.log_level)
    setup_logging(
        log_level=log_level,
        log_file=args.log_file
    )
    
    try:
        logger.info("Initializing experiment runner")
        
        # Load configurations
        logger.info(f"Loading experiment configuration: {args.config}")
        experiment_config: Dict[str, Any] = load_experiment_config(args.config)
        
        logger.info("Loading node configuration")
        node_config: Dict[str, Dict[str, Any]] = load_node_config()
        
        # Initialize node controller
        logger.info("Initializing node controller")
        node_controller: NodeController = NodeController(node_config)
        
        # Initialize experiment runner
        logger.info("Initializing experiment runner")
        runner: ExperimentRunner = ExperimentRunner(
            node_controller=node_controller,
            config=experiment_config,
            host=args.host,
            port=args.port,
            experiment_id=args.experiment_id,
            run_id=args.run_id,
            data_file=args.data_file
        )
        
        # Run experiment
        logger.info("Starting experiment")
        success: bool = runner.run()
        
        if success:
            logger.info("Experiment completed successfully")
            return 0
        else:
            logger.error("Experiment failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        return 130
        
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
