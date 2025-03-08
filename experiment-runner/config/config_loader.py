#!/usr/bin/env python3
"""
Configuration loader for experiment runner.

Loads and validates configuration files for experiments and nodes.
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

def load_experiment_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Load experiment configuration.
    
    Args:
        config_name: Name of the configuration to load (default or test)
        
    Returns:
        Dictionary containing the experiment configuration
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine configuration file path
    if config_name == "default":
        config_path = os.path.join(base_dir, "default.yaml")
    elif config_name == "test":
        config_path = os.path.join(base_dir, "test.yaml")
    else:
        raise ValueError(f"Invalid configuration name: {config_name}")
        
    # Load the configuration
    config = load_yaml_config(config_path)
    
    # Process the configuration to expand ranges
    processed_config = process_experiment_config(config)
    
    return processed_config

def process_experiment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process experiment configuration to expand ranges into explicit values.
    
    Args:
        config: Raw experiment configuration
        
    Returns:
        Processed configuration with expanded ranges
    """
    processed_config: Dict[str, Any] = {
        "description": config.get("description", ""),
        "dwell_time_s": config.get("dwell_time_s", 5),
    }
    
    # Expand prb_configs range
    prb_config: Dict[str, Any] = config.get("prb_configs", {})
    min_prb_range: Dict[str, int] = prb_config.get("min_prb_ratio_range", {})
    start: int = min_prb_range.get("start", 50)
    end: int = min_prb_range.get("end", 96)
    step: int = min_prb_range.get("step", 1)
    
    prb_configs: List[Dict[str, int]] = [
        {
            "ue_id": prb_config.get("ue_id", 1),
            "sst": prb_config.get("sst", 1),
            "sd": prb_config.get("sd", 1),
            "min_prb_ratio": min_prb,
            "max_prb_ratio": prb_config.get("max_prb_ratio", 100)
        }
        for min_prb in range(start, end + 1, step)
    ]
    
    # Expand attens range
    attens_range: Dict[str, int] = config.get("attens_range", {})
    start = attens_range.get("start", 10)
    end = attens_range.get("end", 37)
    step = attens_range.get("step", 1)
    
    attens: List[int] = [i for i in range(start, end + 1, step)]
    
    processed_config["prb_configs"] = prb_configs
    processed_config["attens"] = attens
    
    return processed_config

def load_node_config() -> Dict[str, Any]:
    """
    Load node configuration.
    
    Returns:
        Dictionary containing the node configuration
    """
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
