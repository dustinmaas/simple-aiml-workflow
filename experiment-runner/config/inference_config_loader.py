#!/usr/bin/env python3
"""
Inference configuration loader for experiment runner.

Loads and validates configuration files for inference experiments.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the file does not exist
        yaml.YAMLError: If there is an error parsing the YAML
    """
    logger.info(f"Loading configuration from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            logger.debug(f"Loaded configuration: {config}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise

def load_inference_config(config_path: str) -> Dict[str, Any]:
    """
    Load inference experiment configuration.
    
    Args:
        config_path: Path to the inference configuration file
        
    Returns:
        Dictionary containing the processed inference configuration
    """
    # Load the configuration
    config = load_yaml_config(config_path)
    
    # Validate required fields
    required_fields = ['description', 'attens_and_target_thps', 'dwell_time_s']
    for field in required_fields:
        if field not in config:
            logger.warning(f"Missing required field '{field}' in inference configuration")
    
    return config

def load_node_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load node configuration from the specified path or default location.
    
    Args:
        config_path: Optional path to the node configuration file
        
    Returns:
        Dictionary containing the processed node configuration
    """
    # Determine configuration path
    if config_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "node_config.yaml")
    
    # Load raw config
    raw_config = load_yaml_config(config_path)
    
    # Process the configuration to substitute constants in command strings
    return process_node_config(raw_config)

def process_node_config(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Process node configuration to substitute constants in command strings.
    
    Args:
        config: Raw node configuration
        
    Returns:
        Processed node configuration with constants substituted
    """
    # Extract constants
    constants: Dict[str, str] = config.get("constants", {})
    
    # Process nodes and substitute constants in commands
    nodes_config: Dict[str, Dict[str, Any]] = {}
    
    for node_name, node_info in config.get("nodes", {}).items():
        processed_commands: Dict[str, str] = {}
        
        for cmd_name, cmd_template in node_info.get("commands", {}).items():
            # Format command template with constants
            processed_cmd = cmd_template.format(**constants)
            processed_commands[cmd_name] = processed_cmd
        
        # Create processed node config
        nodes_config[node_name] = {
            "hostname": node_info.get("hostname"),
            "commands": processed_commands
        }
    
    return nodes_config
