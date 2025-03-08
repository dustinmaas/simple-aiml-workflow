#!/usr/bin/env python3
"""
Experiment runner module.

Handles the setup, execution, and cleanup of experiments.
"""

import os
import json
import time
import uuid
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
import requests
from requests.exceptions import RequestException
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

from core.node_controller import NodeController
from core.signal_handler import SignalHandler

logger = logging.getLogger(__name__)

# Constants
DEFAULT_HOST = "10.0.2.1"
DEFAULT_PORT = 61611
DEFAULT_ATTEN_SCRIPT = "/local/repository/bin/update-attens"
UE_ID_TO_MATRIX_ID = {1: 1, 3: 2}

# InfluxDB Constants
DEFAULT_INFLUXDB_URL = "http://localhost:8086"
DEFAULT_INFLUXDB_TOKEN = "ric_admin_token"
DEFAULT_INFLUXDB_ORG = "ric"
DEFAULT_INFLUXDB_BUCKET = "network_metrics"
DEFAULT_INFLUXDB_TAGS = [
    "ue_id",
    "sst",
    "sd",
    "min_prb_ratio",
    "max_prb_ratio",
    "atten"
]

class ExperimentRunner:
    """
    Runner for network experiments.
    
    Sets up network components, runs experiments, and collects measurements.
    """
    
    def __init__(
        self,
        node_controller: NodeController,
        config: Dict[str, Any],
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        data_file: str = "data.txt"
    ):
        """
        Initialize experiment runner.
        
        Args:
            node_controller: NodeController instance for remote command execution
            config: Experiment configuration
            host: API server host
            port: API server port
            experiment_id: ID for the experiment (defaults to timestamp-based ID)
            run_id: ID for this particular run (defaults to UUID-based ID)
            data_file: Path to file for storing measurement data
        """
        self.node_controller = node_controller
        self.config = config
        self.base_url = f"http://{host}:{port}"
        self.data_file = data_file
        
        # Generate experiment and run IDs if not provided
        self.experiment_id = experiment_id or f"exp_{int(time.time())}"
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        
        # Initialize state variables
        self.prb_ratios = {}
        self.attens = {}
        self.running = False
        
        # Initialize InfluxDB client attributes
        self.influxdb_url = DEFAULT_INFLUXDB_URL
        self.influxdb_token = DEFAULT_INFLUXDB_TOKEN
        self.influxdb_org = DEFAULT_INFLUXDB_ORG
        self.influxdb_bucket = DEFAULT_INFLUXDB_BUCKET
        self.influxdb_tags = DEFAULT_INFLUXDB_TAGS
        self.influxdb_client = None
        self.influxdb_write_api = None
        
        # Initialize signal handler
        self.signal_handler = SignalHandler()
        self.signal_handler.register_cleanup_function(self._cleanup)
        self.signal_handler.setup_signal_handlers()
        
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Run ID: {self.run_id}")
        
    def _update_atten(self, atten: int, ue: int = 1) -> None:
        """
        Update attenuation for a UE.
        
        Args:
            atten: Attenuation value
            ue: UE identifier
        """
        if ue not in UE_ID_TO_MATRIX_ID:
            logger.error(f"Invalid UE ID: {ue}")
            return
            
        matrix_id = UE_ID_TO_MATRIX_ID[ue]
        cmd = f"{DEFAULT_ATTEN_SCRIPT} ru1ue{matrix_id} {atten}"
        
        logger.info(f"Setting attenuation for UE {ue} to {atten}")
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Error updating attenuation: {result.stderr}")
        else:
            self.attens[ue] = atten
            logger.info(f"Updated attenuation for UE {ue} to {atten}")

    def _update_prb_ratio(
        self,
        ue_id: int,
        sst: int,
        sd: int,
        min_prb_ratio: int,
        max_prb_ratio: int
    ) -> None:
        """
        Update PRB ratio for a UE.
        
        Args:
            ue_id: UE identifier
            sst: Slice type
            sd: Slice differentiator
            min_prb_ratio: Minimum PRB ratio
            max_prb_ratio: Maximum PRB ratio
        """
        try:
            logger.info(f"Updating PRB ratio for UE {ue_id}: min={min_prb_ratio}, max={max_prb_ratio}")
            
            response = requests.post(
                f"{self.base_url}/update_prb_ratio",
                json={
                    "ue_id": ue_id,
                    "sst": sst,
                    "sd": sd,
                    "min_prb_ratio": min_prb_ratio,
                    "max_prb_ratio": max_prb_ratio,
                },
                timeout=10
            )
            response.raise_for_status()
            
            # Update internal state
            if ue_id not in self.prb_ratios:
                self.prb_ratios[ue_id] = {}
                
            self.prb_ratios[ue_id]["min_prb_ratio"] = min_prb_ratio
            self.prb_ratios[ue_id]["max_prb_ratio"] = max_prb_ratio
            self.prb_ratios[ue_id]["sst"] = sst
            self.prb_ratios[ue_id]["sd"] = sd
            
            logger.info(f"Updated PRB ratio for UE {ue_id}: min={min_prb_ratio}, max={max_prb_ratio}")
            
        except RequestException as e:
            logger.error(f"Error updating PRB ratio for UE {ue_id}: {e}")

    def _check_xapp_api(self, max_retries: int = 30, retry_interval: int = 1) -> bool:
        """
        Check if xApp API is accessible.
        
        Args:
            max_retries: Maximum number of retries
            retry_interval: Time between retries in seconds
            
        Returns:
            bool: True if API is accessible, False otherwise
        """
        logger.info("Checking xApp API availability...")
        
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("xApp API is available")
                    return True
            except Exception as e:
                logger.debug(f"Attempt {i+1}/{max_retries}: API not ready yet: {e}")
                
            time.sleep(retry_interval)
            
        logger.error("xApp API is not available after maximum retries")
        return False
        
    def _setup_experiment(self) -> bool:
        """
        Set up experiment by starting all components in the correct order.
        
        Returns:
            bool: True if setup succeeded, False otherwise
        """
        logger.info("Setting up experiment...")

        # Stop everything first
        logger.info("Stopping all services...")
        self.node_controller.execute_command(node_name="ue1", command_name="stop_iperf_client")
        self.node_controller.execute_command(node_name="ue3", command_name="stop_iperf_client")
        self.node_controller.execute_command(node_name="ue1", command_name="airplane_mode")
        self.node_controller.execute_command(node_name="ue3", command_name="airplane_mode")
        self.node_controller.execute_command(node_name="cn5g", command_name="stop_iperf_server_1")
        self.node_controller.execute_command(node_name="cn5g", command_name="stop_iperf_server_2")
        self.node_controller.execute_command(node_name="gnb", command_name="stop_gnb")
        self.node_controller.execute_command(node_name="ric", command_name="stop_ric")
        self.node_controller.execute_command(node_name="metrics_influxdb", command_name="stop_influxdb")

        # Start core network and iperf servers
        logger.info("Starting core network and iperf servers...")
        self.node_controller.execute_command(node_name="cn5g", command_name="restart_cn")
        self.node_controller.execute_command(node_name="cn5g", command_name="start_iperf_server_1")
        self.node_controller.execute_command(node_name="cn5g", command_name="start_iperf_server_2")
        time.sleep(5)
        
        # Initialize attenuations
        self.attens = {1: 0, 3: 0}
        self._update_atten(0, ue=1)
        self._update_atten(0, ue=3)

        # Start RIC
        logger.info("Starting RIC...")
        self.node_controller.execute_command(node_name="ric", command_name="start_ric")
        retry_count = 0
        max_retries = 30
        while not self.node_controller.execute_command(node_name="ric", command_name="check_status"):
            retry_count += 1
            if retry_count >= max_retries:
                logger.error("Failed to start RIC after maximum retries")
                return False
            time.sleep(1)
        
        # Start gNB
        logger.info("Starting gNB...")
        self.node_controller.execute_command(node_name="gnb", command_name="start_gnb")
        retry_count = 0
        while not self.node_controller.execute_command(node_name="gnb", command_name="check_status"):
            retry_count += 1
            if retry_count >= max_retries:
                logger.error("Failed to start gNB after maximum retries")
                return False
            time.sleep(1)
        
        # Start UE1 and wait for connection
        logger.info("Starting UE1...")
        self.node_controller.execute_command(node_name="ue1", command_name="online_mode")
        retry_count = 0
        while not self.node_controller.execute_command(node_name="ue1", command_name="check_status"):
            retry_count += 1
            if retry_count >= max_retries:
                logger.error("Failed to connect UE1 after maximum retries")
                return False
            time.sleep(1)
            
        # Start UE3 and wait for connection
        logger.info("Starting UE3...")
        self.node_controller.execute_command(node_name="ue3", command_name="online_mode")
        retry_count = 0
        while not self.node_controller.execute_command(node_name="ue3", command_name="check_status"):
            retry_count += 1
            if retry_count >= max_retries:
                logger.error("Failed to connect UE3 after maximum retries")
                return False
            time.sleep(1)

        # Start iperf clients
        logger.info("Starting iperf clients...")
        self.node_controller.execute_command(node_name="ue1", command_name="start_iperf_client")
        self.node_controller.execute_command(node_name="ue3", command_name="start_iperf_client")
        
        # Start InfluxDB
        logger.info("Starting InfluxDB...")
        self.node_controller.execute_command(node_name="metrics_influxdb", command_name="start_influxdb")
        retry_count = 0
        while not self._check_influxdb_running():
            retry_count += 1
            if retry_count >= max_retries:
                logger.error("Failed to start InfluxDB after maximum retries")
                return False
            time.sleep(1)
        
        # Connect to InfluxDB
        self._connect_to_influxdb()
        
        # Start AIML xApp and wait for API
        logger.info("Starting AIML xApp...")
        self.node_controller.execute_command(node_name="aiml_xapp", command_name="start_xapp")
        retry_count = 0
        while not self.node_controller.execute_command(node_name="aiml_xapp", command_name="check_status"):
            retry_count += 1
            if retry_count >= max_retries:
                logger.error("Failed to start AIML xApp after maximum retries")
                return False
            time.sleep(1)
        
        # Check xApp API
        if not self._check_xapp_api():
            logger.error("xApp API is not available")
            return False
        
        # Initialize PRB ratios
        self.prb_ratios = {
            1: {"min_prb_ratio": 50, "max_prb_ratio": 100, "sst": 1, "sd": 1},
            3: {"min_prb_ratio": 0, "max_prb_ratio": 100, "sst": 1, "sd": 2}
        }
        self._update_prb_ratio(1, 1, 1, 50, 100)
        self._update_prb_ratio(3, 1, 2, 0, 100)
        
        logger.info("Experiment setup complete")
        return True
        
    def _check_influxdb_running(self) -> bool:
        """
        Check if InfluxDB is running.
        
        Returns:
            bool: True if InfluxDB is running, False otherwise
        """
        try:
            response = requests.get("http://localhost:8086/health", timeout=2)
            if response.status_code == 200:
                logger.info("InfluxDB is running")
                return True
            else:
                logger.warning(f"InfluxDB health check returned status code: {response.status_code}")
                return False
        except Exception as e:
            logger.debug(f"Error checking InfluxDB status: {e}")
            return False
            
    def _collect_measurements(self) -> Dict[str, Dict[str, Any]]:
        """
        Collect measurements from the xApp API.
        
        Returns:
            Dictionary of measurements indexed by UE ID
        """
        try:
            response = requests.get(f"{self.base_url}/measurements", timeout=5)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"Error collecting measurements: {e}")
            return {}
            
    def _connect_to_influxdb(self) -> bool:
        """
        Connect to InfluxDB.
        
        Returns:
            True if connection succeeded, False otherwise
        """
        try:
            logger.info(f"Connecting to InfluxDB at {self.influxdb_url}")
            self.influxdb_client = influxdb_client.InfluxDBClient(
                url=self.influxdb_url,
                token=self.influxdb_token,
                org=self.influxdb_org
            )
            self.influxdb_write_api = self.influxdb_client.write_api(write_options=SYNCHRONOUS)
            logger.info("Connected to InfluxDB")
            return True
        except Exception as e:
            logger.error(f"Error connecting to InfluxDB: {e}")
            return False
            
    def _close_influxdb(self) -> None:
        """
        Close connection to InfluxDB.
        """
        if self.influxdb_client:
            try:
                if self.influxdb_write_api:
                    self.influxdb_write_api.close()
                self.influxdb_client.close()
                logger.info("Closed connection to InfluxDB")
            except Exception as e:
                logger.error(f"Error closing InfluxDB connection: {e}")
                
    def _store_influxdb_measurement(
        self,
        measurement_dict: Dict[str, Any],
        measurement_name: str = "network_metrics"
    ) -> bool:
        """
        Store measurement data in InfluxDB.
        
        Args:
            measurement_dict: Dictionary containing measurement data
            measurement_name: Name of the measurement
            
        Returns:
            True if storage succeeded, False otherwise
        """
        if not self.influxdb_client or not self.influxdb_write_api:
            logger.error("Not connected to InfluxDB")
            return False
            
        try:
            # Create a point with tags and fields
            point = influxdb_client.Point(measurement_name)
            
            # Add experiment and run ID tags
            point = point.tag("experiment_id", self.experiment_id)
            point = point.tag("run_id", self.run_id)
            
            # Add custom tags from measurement data
            for tag_name in self.influxdb_tags:
                if tag_name in measurement_dict:
                    tag_value = str(measurement_dict[tag_name])
                    point = point.tag(tag_name, tag_value)
            
            # Add all fields (excluding tags)
            for key, value in measurement_dict.items():
                if key not in self.influxdb_tags:
                    # Convert value to float if numeric
                    try:
                        value_float = float(value)
                        point = point.field(key, value_float)
                    except (ValueError, TypeError):
                        if isinstance(value, str):
                            point = point.field(key, value)
            
            # Write to InfluxDB
            self.influxdb_write_api.write(bucket=self.influxdb_bucket, record=point)
            logger.info(f"Stored measurement in InfluxDB for UE {measurement_dict.get('ue_id', '')}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing data in InfluxDB: {e}")
            return False
    
    def _store_measurement(
        self,
        ue_id: int,
        data: Dict[str, Any]
    ) -> None:
        """
        Store a measurement.
        
        Args:
            ue_id: UE identifier
            data: Measurement data
        """
        # Skip if UE doesn't have PRB settings
        if ue_id not in self.prb_ratios:
            logger.warning(f"No PRB settings for UE {ue_id}, skipping")
            return
            
        # Skip if UE doesn't have attenuation settings
        if ue_id not in self.attens:
            logger.warning(f"No attenuation settings for UE {ue_id}, skipping")
            return
        
        # Create measurement dictionary
        measurement_dict = {
            "timestamp": datetime.now().isoformat(),
            "collect_start_time": data.get("colletStartTime"),  # Original API has typo
            "ue_id": ue_id,
            "sst": self.prb_ratios[ue_id].get("sst", 1),
            "sd": self.prb_ratios[ue_id].get("sd", 1),
            "min_prb_ratio": self.prb_ratios[ue_id]["min_prb_ratio"],
            "max_prb_ratio": self.prb_ratios[ue_id]["max_prb_ratio"],
            "atten": self.attens[ue_id]
        }

        # Add measurement data
        meas_data = data.get("measData", {})
        if not meas_data:
            logger.warning(f"No measurement data for UE {ue_id}")
        
        for key, value in meas_data.items():
            measurement_dict[key] = value[0] if isinstance(value, (list, tuple)) else value

        # Store in file
        with open(self.data_file, "a") as f:
            f.write(json.dumps(measurement_dict) + "\n")
            
        # Store in InfluxDB
        self._store_influxdb_measurement(measurement_dict)
        
        logger.info(f"Stored measurement for UE {ue_id} (Atten: {measurement_dict['atten']}, Min PRB: {measurement_dict['min_prb_ratio']})")
            
    def _cleanup(self) -> None:
        """
        Clean up resources and stop nodes.
        """
        logger.info("Cleaning up experiment resources...")
        
        # Save docker logs
        logger.info("Saving docker logs...")
        self.node_controller.execute_command(node_name="ric", command_name="get_docker_logs")
        
        # Stop nodes
        self.node_controller.stop_nodes()
        
        # Close InfluxDB connection
        self._close_influxdb()
        
    def run(self) -> bool:
        """
        Run the experiment.
        
        Returns:
            bool: True if experiment succeeded, False otherwise
        """
        logger.info("Starting experiment...")
        self.running = True
        
        try:
            # Set up experiment components
            if not self._setup_experiment():
                logger.error("Failed to set up experiment")
                return False
                
            # Create data file
            with open(self.data_file, "w") as f:
                # Empty file
                pass
                
            # Run experiment
            for atten in self.config["attens"]:
                logger.info(f"Setting attenuation to {atten}")
                self._update_atten(atten, ue=1)
                logger.info(f"Current attenuation values: {{{', '.join(f'{k}: {v}' for k, v in self.attens.items())}}}")

                # Iterate through each PRB configuration
                for prb_config in self.config["prb_configs"]:
                    ue_id = prb_config["ue_id"]
                    logger.info(f"Configuring UE {ue_id} with {prb_config}")
                    
                    self._update_prb_ratio(
                        ue_id=ue_id,
                        sst=prb_config["sst"],
                        sd=prb_config["sd"],
                        min_prb_ratio=prb_config["min_prb_ratio"],
                        max_prb_ratio=prb_config["max_prb_ratio"]
                    )
                    
                    # Collect measurements for duration `dwell_time_s`
                    measurements_collected = 0
                    for iteration in range(self.config["dwell_time_s"]):
                        logger.info(f"Collecting measurements (iteration {iteration+1}/{self.config['dwell_time_s']})")
                        
                        measurements = self._collect_measurements()
                        
                        if not measurements:
                            logger.warning("Received empty measurements dictionary")
                            time.sleep(1)
                            continue
                        
                        logger.info(f"Received measurements for UEs: {list(measurements.keys())}")
                        
                        for ue_id_str, data in measurements.items():
                            ue_id = int(ue_id_str)
                            self._store_measurement(ue_id, data)
                            measurements_collected += 1

                        time.sleep(1)
                        
                    logger.info(f"Collected {measurements_collected} measurements for this configuration")

            logger.info("Experiment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in experiment run: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            self.running = False
            self._cleanup()
