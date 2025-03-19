#!/usr/bin/env python3
"""
Node controller module.

Handles remote execution of commands on network nodes via SSH.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from fabric import Connection
from fabric.runners import Result

logger = logging.getLogger(__name__)

class NodeController:
    """
    Controller for managing remote nodes via fabric.
    
    This class establishes SSH connections to remote nodes and executes commands.
    """
    
    def __init__(self, config: Dict[str, Dict[str, Any]]):
        """
        Initialize node controller.
        
        Args:
            config: Node configuration dictionary, containing hostname and commands for each node
        """
        self.config: Dict[str, Dict[str, Any]] = config
        self.connections: Dict[str, Connection] = {}
        self._setup_connections()
        
    def _setup_connections(self) -> None:
        """
        Set up SSH connections to nodes.
        
        Establishes SSH connections to all configured nodes.
        """
        for node_name, node_info in self.config.items():
            try:
                conn = Connection(host=node_info["hostname"])
                self.connections[node_name] = conn
                logger.info(f"Connected to {node_name} node ({node_info['hostname']})")
            except Exception as e:
                logger.error(f"Failed to connect to {node_name} node: {e}")
                
    def execute_command(
        self, 
        node_name: str, 
        command: str = None, 
        command_name: str = None,
        return_result: bool = False
    ) -> Union[bool, Optional[Result]]:
        """
        Execute a command on a node.
        
        This method can run either:
          - A raw command string provided directly
          - A predefined command looked up from the configuration
        
        Args:
            node_name: Name of node (e.g., 'ue1', 'gnb', 'ric')
            command: Raw command string to execute (if provided)
            command_name: Name of command to run from configuration (if command is None)
            return_result: If True, returns the full Result object; if False, returns a boolean success flag
            
        Returns:
            If return_result=True: Command result object or None if execution failed
            If return_result=False: True if command succeeded (return code 0), False otherwise
            
        Raises:
            ValueError: If neither command nor command_name is provided
            KeyError: If command_name is not found in configuration
        """
        if node_name not in self.connections:
            logger.error(f"No connection to {node_name} node")
            return False if not return_result else None
        
        # Determine the command to run
        cmd_to_execute = None
        if command is not None:
            cmd_to_execute = command
        elif command_name is not None:
            try:
                if command_name not in self.config[node_name]["commands"]:
                    raise KeyError(f"Command {command_name} not found for node {node_name}")
                cmd_to_execute = self.config[node_name]["commands"][command_name]
            except Exception as e:
                logger.error(f"Error looking up command '{command_name}' for node '{node_name}': {e}")
                return False if not return_result else None
        else:
            error_msg = "Either command or command_name must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Execute the command
        try:
            logger.info(f"Executing on {node_name}: {cmd_to_execute}")
            
            # Capture output with hide=True
            result = self.connections[node_name].run(cmd_to_execute, hide=True, warn=True)
            logger.info(f"Command completed with exit code: {result.return_code}")
            
            # Log stdout with timestamp
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        if "Container" in line and any(action in line for action in ["Creating", "Starting", "Stopping", "Removing"]):
                            # Format Docker Compose output with timestamp
                            logger.info(f"DOCKER: {line.strip()}")
                        else:
                            logger.debug(f"STDOUT: {line.strip()}")
            
            # Log stderr with timestamp
            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    if line.strip():
                        # Check different types of output
                        if "Container" in line and any(action in line for action in ["Creating", "Starting", "Stopping", "Removing", "Stopped", "Created", "Removed", "Started"]):
                            # Format Docker Compose output with timestamp
                            logger.info(f"DOCKER: {line.strip()}")
                        elif "Network" in line:
                            # Network messages from Docker Compose
                            logger.info(f"DOCKER: {line.strip()}")
                        elif any(chat_msg in line for chat_msg in ["send", "expect", "got it", "AT", "OK", "^M"]):
                            # Chat command output - prefix with UE ID if this is a UE node
                            ue_prefix = ""
                            if node_name.startswith("ue"):
                                ue_prefix = f"{node_name}: "
                            logger.info(f"CHAT: {ue_prefix}{line.strip()}")
                        elif "service" in line and "is not running" in line:
                            # Docker service status messages
                            logger.info(f"DOCKER: {line.strip()}")
                        elif "Copying" in line or "Copied" in line:
                            # Docker copy operations
                            logger.info(f"DOCKER: {line.strip()}")
                        else:
                            logger.warning(f"STDERR: {line.strip()}")
            
            # Return appropriate result based on return_result flag
            if return_result:
                return result
            else:
                return result.return_code == 0
            
        except Exception as e:
            cmd_type = "command" if command else f"command_name '{command_name}'"
            logger.error(f"Error executing {cmd_type} on {node_name}: {e}")
            return False if not return_result else None
            
    # Maintain backwards compatibility
    def run_configured_command(self, node_name: str, command_name: str) -> bool:
        """
        Backwards compatibility method for running a predefined command.
        
        Args:
            node_name: Name of node (e.g., 'ue1', 'gnb', 'ric')
            command_name: Name of command to run as defined in configuration
            
        Returns:
            True if command succeeded (return code 0), False otherwise
        """
        return self.execute_command(node_name=node_name, command_name=command_name)
            
    def stop_nodes(self) -> None:
        """
        Stop all nodes in reverse order: UEs -> gNB -> CN -> RIC.
        
        This ensures a clean shutdown of the system.
        """
        logger.info("Stopping all nodes...")
        
        # Stop UEs
        self.execute_command(node_name="ue1", command_name="stop_iperf_client")
        self.execute_command(node_name="ue1", command_name="airplane_mode")
        self.execute_command(node_name="ue3", command_name="stop_iperf_client")
        self.execute_command(node_name="ue3", command_name="airplane_mode")
        
        # Stop gNB
        self.execute_command(node_name="gnb", command_name="stop_gnb")
        
        # Stop core network
        self.execute_command(node_name="cn5g", command_name="stop_iperf_server_1")
        self.execute_command(node_name="cn5g", command_name="stop_iperf_server_2")
        
        # Stop RIC and related services - xApp is stopped when RIC is stopped
        self.execute_command(node_name="ric", command_name="stop_ric")
        self.execute_command(node_name="datalake_influxdb", command_name="stop_influxdb")
        
        logger.info("All nodes stopped")

    def check_status(self) -> Dict[str, bool]:
        """
        Check status of all nodes.
        
        Returns:
            Dictionary with node names as keys and status (True/False) as values
        """
        logger.info("Checking node status...")
        
        status = {
            "cn5g": self.execute_command(node_name="cn5g", command_name="check_status"),
            "gnb": self.execute_command(node_name="gnb", command_name="check_status"),
            "ue1": self.execute_command(node_name="ue1", command_name="check_status"),
            "ue3": self.execute_command(node_name="ue3", command_name="check_status"),
            "ric": self.execute_command(node_name="ric", command_name="check_status"),
            "aiml_xapp": self.execute_command(node_name="aiml_xapp", command_name="check_status"),
        }
        
        return status

    def __del__(self) -> None:
        """
        Clean up connections on object destruction.
        
        Ensures that all SSH connections are properly closed.
        """
        logger.info("Cleaning up node connections...")
        
        for conn in self.connections.values():
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
