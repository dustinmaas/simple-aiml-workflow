#!/usr/bin/env python3
"""
Runner script for OSC RIC experiment.

This module implements an experiment runner that configures PRB ratios and
collects measurements through the xApp REST API.
"""

# Standard library imports
import argparse
import json
import subprocess
import time
from datetime import datetime
import signal
import sys
import uuid

# Third-party imports
import requests
from requests.exceptions import RequestException
from fabric import Connection
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

# Constants
SRSRAN_PATH = "/var/tmp/srsRAN_Project"
CORE_NETWORK_IP = "10.45.0.1"
RIC_PATH = "/var/tmp/simple-aiml-workflow/oran-sc-ric"
UE_ID_TO_MATRIX_ID = {1: 1, 3: 2}
DEFAULT_HOST = "10.0.2.1"
DEFAULT_PORT = 61611
DEFAULT_ATTEN_SCRIPT = "/local/repository/bin/update-attens"

# InfluxDB Configuration
INFLUXDB_URL = "http://10.0.2.25:8086"
INFLUXDB_TOKEN = "ric_admin_token"
INFLUXDB_ORG = "ric"
INFLUXDB_BUCKET = "network_metrics"
INFLUXDB_TAGS = [
    "ue_id",
    "sst",
    "sd",
    "min_prb_ratio",
    "max_prb_ratio",
    "atten"
]

# ML Inference Server Configuration
ML_INFERENCE_URL = "http://10.0.2.30:8888"

DEFAULT_EXP_CONFIG = {
    "description": """This experiment tests the performance of the system under 
    different attenuation values and PRB ratios for UE 1 slice.""",
    "prb_configs": [
        {
            "ue_id": 1,
            "sst": 1,
            "sd": 1,
            "min_prb_ratio": min_prb,
            "max_prb_ratio": 100
        }
        for min_prb in range(50, 96, 1)
    ],
    "attens": [i for i in range(10, 37)],
    "dwell_time_s": 5
}

TEST_EXP_CONFIG = {
    "description": """This experiment tests the performance of the system under 
    different attenuation values and PRB ratios for UE 1 slice.""",
    "prb_configs": [
        {
            "ue_id": 1,
            "sst": 1,
            "sd": 1,
            "min_prb_ratio": min_prb,
            "max_prb_ratio": 100
        }
        for min_prb in range(50, 100, 20)
    ],
    "attens": [i for i in range(10, 37, 5)],
    "dwell_time_s": 2
}

# Node configuration
NODE_CONFIG = {
    "ue1": {
        "hostname": "nuc27",
        "commands": {
            "restart_con_manager": "sudo systemctl restart quectel-cm",
            "airplane_mode": """sudo sh -c \"chat -t 1 -sv '' AT OK 'AT+CFUN=4' OK < /dev/ttyUSB2 > /dev/ttyUSB2\"""",
            "online_mode": """sudo sh -c \"chat -t 1 -sv '' AT OK 'AT+CFUN=1' OK < /dev/ttyUSB2 > /dev/ttyUSB2\"""",
            "start_iperf_client": "nohup iperf3 -c " + CORE_NETWORK_IP + " -p 5201 -t10000 -u -b300M -R > /tmp/iperf3_ue1.log 2>&1 &",
            "stop_iperf_client": "sudo pkill -SIGINT -f iperf3",
            "check_status": "ping -c 1 -W 1 -w 1 " + CORE_NETWORK_IP
        }
    },
    "ue3": {
        "hostname": "nuc22",
        "commands": {
            "restart_con_manager": "sudo systemctl restart quectel-cm",
            "airplane_mode": """sudo sh -c \"chat -t 1 -sv '' AT OK 'AT+CFUN=4' OK < /dev/ttyUSB2 > /dev/ttyUSB2\"""",
            "online_mode": """sudo sh -c \"chat -t 1 -sv '' AT OK 'AT+CFUN=1' OK < /dev/ttyUSB2 > /dev/ttyUSB2\"""",
            "start_iperf_client": "nohup iperf3 -c " + CORE_NETWORK_IP + " -p 5202 -t10000 -u -b300M -R > /tmp/iperf3_ue3.log 2>&1 &",
            "stop_iperf_client": "sudo pkill -SIGINT -f iperf3",
            "check_status": "ping -c 1 -W 1 -w 1 " + CORE_NETWORK_IP
        }
    },
    "gnb": {
        "hostname": "cudu",
        "commands": {
            "start_gnb": "nohup sudo numactl --membind 1 --cpubind 1 " + SRSRAN_PATH + "/build/apps/gnb/gnb -c /var/tmp/etc/srsran/gnb_rf_x310_ric.yml cell_cfg --channel_bandwidth_MHz 80 > /tmp/gnb-std.log 2>&1 &",
            "stop_gnb": "sudo pkill -SIGINT gnb",
            "check_status": "ps -e | grep gnb"
        }
    },
    "cn5g": {
        "hostname": "cn5g",
        "commands": {
            "restart_cn": "sudo systemctl restart open5gs-*",
            "start_iperf_server_1": "nohup iperf3 -s -p 5201 > /tmp/iperf3_server1.log 2>&1 &",
            "start_iperf_server_2": "nohup iperf3 -s -p 5202 > /tmp/iperf3_server2.log 2>&1 &",
            "stop_iperf_server_1": "pkill -SIGINT -f 'iperf3 -s -p 5201'",
            "stop_iperf_server_2": "pkill -SIGINT -f 'iperf3 -s -p 5202'",
            "check_status": "sudo systemctl status open5gs-*"
        }
    },
    "ric": {
        "hostname": "ric",
        "commands": {
            "start_ric": "cd " + RIC_PATH + " && sudo docker compose up -d",
            "stop_ric": "cd " + RIC_PATH + " && sudo docker compose down",
            "get_docker_logs": "cd " + RIC_PATH + " && sudo docker compose logs > /tmp/final_docker_logs.log",
            "check_status": "cd " + RIC_PATH + " && sudo docker compose ps"
        }
    },
    "aiml_xapp": {
        "hostname": "ric",
        "commands": {
            "start_xapp": "cd " + RIC_PATH + " && sudo docker compose cp /var/tmp/simple-aiml-workflow/aiml_xapp.py python_xapp_runner:/opt/xApps/ && sudo docker compose exec -d python_xapp_runner python3 /opt/xApps/aiml_xapp.py --ue_ids=1,3",
            "stop_xapp": "cd " + RIC_PATH + " && sudo docker compose exec python_xapp_runner pkill -f aiml_xapp.py"
        }
    }
}

class NodeController:
    """Controller for managing remote nodes via fabric."""
    
    def __init__(self, config=NODE_CONFIG):
        """Initialize node controller.
        
        Args:
            config: Node configuration dictionary
        """
        self.config = config
        self.connections = {}
        self._setup_connections()
        
    def _setup_connections(self):
        """Set up SSH connections to nodes."""
        for node_name, node_info in self.config.items():
            try:
                conn = Connection(host=node_info["hostname"])
                self.connections[node_name] = conn
                print(f"Connected to {node_name} node")
            except Exception as e:
                print(f"Failed to connect to {node_name} node: {e}")
                
    def _run_command(self, node_name, command_name):
        """Run a command on a node.
        
        Args:
            node_name: Name of node ('ue1', 'ue3', 'gnb', or 'cn5g')
            command_name: Name of command to run
            
        Returns:
            Result of command execution
        """
        if node_name not in self.connections:
            print(f"No connection to {node_name} node")
            return False
            
        try:
            command = self.config[node_name]["commands"][command_name]
            print(f"\nExecuting on {node_name}: {command}")
            result = self.connections[node_name].run(command, hide=False, warn=True)
            print(f"Command completed with exit code: {result.return_code}")
            if result.stdout:
                print(f"STDOUT:\n{result.stdout.strip()}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr.strip()}")
            return result.return_code == 0
        except Exception as e:
            print(f"Error running {command_name} on {node_name}: {e}")
            return False
            
    def start_nodes(self):
        """Start all nodes in correct order: CN -> gNB -> UEs."""
        print("Starting nodes...")
        
        # Start CN first
        cn_started = self._run_command("cn5g", "start_cn")
        if not cn_started:
            return False
        time.sleep(5)  # Wait for CN to initialize
        
        # Start gNB next
        gnb_started = self._run_command("gnb", "start_gnb")
        if not gnb_started:
            self._run_command("cn5g", "stop_cn")  # Cleanup if gNB fails
            return False
        time.sleep(5)  # Wait for gNB to initialize
        
        # Start UEs last
        ue1_started = self._run_command("ue1", "start_ue")
        ue3_started = self._run_command("ue3", "start_ue")
        if not (ue1_started and ue3_started):
            # Cleanup if any UE fails
            self._run_command("gnb", "stop_gnb")
            self._run_command("cn5g", "stop_cn")
            return False
            
        return True
        
    def stop_nodes(self):
        """Stop all nodes in reverse order: UEs -> gNB -> CN."""
        self._run_command("ue1", "stop_iperf_client")
        self._run_command("ue1", "airplane_mode")
        self._run_command("ue3", "stop_iperf_client")
        self._run_command("ue3", "airplane_mode")
        self._run_command("gnb", "stop_gnb")
        self._run_command("cn5g", "stop_iperf_server_1")
        self._run_command("cn5g", "stop_iperf_server_2")
        self._run_command("ric", "stop_ric")
        
    def check_status(self):
        """Check status of all nodes."""
        print("Checking node status...")
        self._run_command("cn5g", "check_status")
        self._run_command("gnb", "check_status")
        self._run_command("ue1", "check_status")
        self._run_command("ue3", "check_status")
        
    def __del__(self):
        """Cleanup connections on deletion."""
        for conn in self.connections.values():
            try:
                conn.close()
            except:
                pass

class ExperimentRunner:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, config=None, 
                experiment_id=None, run_id=None):
        """Initialize experiment runner.
        
        Args:
            host: API server host
            port: API server port
            config: Experiment configuration
            experiment_id: ID for the experiment (defaults to timestamp-based ID)
            run_id: ID for this particular run (defaults to UUID-based ID)
        """
        self.base_url = f"http://{host}:{port}"
        self.config = config or DEFAULT_EXP_CONFIG
        self.running = False
        self.data_file = "data.txt"
        self.prb_ratios = {}
        self.attens = {}
        
        # Generate experiment and run IDs if not provided
        self.experiment_id = experiment_id or f"exp_{int(time.time())}"
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Run ID: {self.run_id}")
        
        # Initialize node controller
        self.node_controller = NodeController()
        
        # Initialize InfluxDB client
        self.influx_client = influxdb_client.InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle signals for graceful shutdown.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        print(f"\nReceived signal {signum}. Performing cleanup...")
        
        # Save docker logs
        print("Saving docker logs...")
        self.node_controller._run_command("ric", "get_docker_logs")
        
        # Stop all nodes
        print("Stopping all nodes...")
        self.node_controller.stop_nodes()
        
        print("Cleanup complete. Exiting...")
        sys.exit(0)

    def _update_atten(self, atten, ue=1):
        """Update attenuation for a UE. UE ids are
        
        Args:
            atten: Attenuation value
            ue: UE identifier
        """
        result = subprocess.run(
            f"/local/repository/bin/update-attens ru1ue{UE_ID_TO_MATRIX_ID[ue]} {atten}",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error updating attenuation: {result.stderr}")
        self.attens[ue] = atten

    def _update_prb_ratio(self, ue_id, sst, sd, min_prb_ratio, max_prb_ratio):
        """Update PRB ratio for a UE.
        
        Args:
            ue_id: UE identifier
            sst: Slice type
            sd: Slice differentiator
            min_prb_ratio: Minimum PRB ratio
            max_prb_ratio: Maximum PRB ratio
        """
        try:
            print(f"\nUpdating PRB ratio for UE {ue_id}: min={min_prb_ratio}, max={max_prb_ratio}")
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
            # Fix the prb_ratios dictionary structure to be consistent
            if ue_id not in self.prb_ratios:
                self.prb_ratios[ue_id] = {}
            self.prb_ratios[ue_id]["min_prb_ratio"] = min_prb_ratio
            self.prb_ratios[ue_id]["max_prb_ratio"] = max_prb_ratio
            self.prb_ratios[ue_id]["sst"] = sst
            self.prb_ratios[ue_id]["sd"] = sd
            print(f"Updated PRB ratio for UE {ue_id}: min={min_prb_ratio}, max={max_prb_ratio}")
            # Verify the update in the dictionary
            print(f"Current PRB settings for UE {ue_id}: {self.prb_ratios[ue_id]}")
        except RequestException as e:
            print(f"Error updating PRB ratio for UE {ue_id}: {e}")

    def _check_xapp_api(self, max_retries=30, retry_interval=1):
        """Check if xApp API is accessible.
        
        Args:
            max_retries: Maximum number of retries
            retry_interval: Time between retries in seconds
            
        Returns:
            bool: True if API is accessible, False otherwise
        """
        print("Checking xApp API availability...")
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    print("xApp API is available")
                    return True
            except Exception as e:
                print(f"Attempt {i+1}/{max_retries}: API not ready yet: {e}")
            time.sleep(retry_interval)
        return False

    def _store_in_influxdb(self, measurement_dict):
        """Store measurement data in InfluxDB.
        
        Args:
            measurement_dict: Dictionary containing measurement data
        """
        try:
            # Create a point with tags and fields
            point = influxdb_client.Point("network_metrics")
            
            # Add tags
            point = point.tag("experiment_id", self.experiment_id)
            point = point.tag("run_id", self.run_id)
            for tag in INFLUXDB_TAGS:
                point = point.tag(tag, str(measurement_dict.get(tag, "")))
            
            # Add all fields
            for key, value in measurement_dict.items():
                if key not in INFLUXDB_TAGS:
                    # Convert value to float if numeric
                    try:
                        value_float = float(value)
                        point = point.field(key, value_float)
                    except (ValueError, TypeError):
                        if isinstance(value, str):
                            point = point.field(key, value)
            
            # Write to InfluxDB
            self.write_api.write(bucket=INFLUXDB_BUCKET, record=point)
            print(f"Stored measurement in InfluxDB for UE {measurement_dict.get('ue_id', '')}")
        except Exception as e:
            print(f"Error storing data in InfluxDB: {e}")
    
    def setup_experiment(self):
        """Set up experiment by starting all components in the correct order."""
        print("Setting up experiment...")

        # Stop everything first
        print("Stopping all services...")
        self.node_controller._run_command("ue1", "stop_iperf_client")
        self.node_controller._run_command("ue3", "stop_iperf_client")
        self.node_controller._run_command("ue1", "airplane_mode")
        self.node_controller._run_command("ue3", "airplane_mode")
        self.node_controller._run_command("cn5g", "stop_iperf_server_1")
        self.node_controller._run_command("cn5g", "stop_iperf_server_2")
        self.node_controller._run_command("aiml_xapp", "stop_xapp")
        self.node_controller._run_command("gnb", "stop_gnb")
        self.node_controller._run_command("ric", "stop_ric")
        
        # Start core network and iperf servers
        print("Starting core network and iperf servers...")
        self.node_controller._run_command("cn5g", "restart_cn")
        self.node_controller._run_command("cn5g", "start_iperf_server_1")
        self.node_controller._run_command("cn5g", "start_iperf_server_2")
        time.sleep(5)
        
        # Initialize attenuations
        self.attens = {1: 0, 3: 0}
        self._update_atten(0, ue=1)
        self._update_atten(0, ue=3)

        # Start RIC
        # import ipdb; ipdb.set_trace()
        print("Starting RIC...")
        self.node_controller._run_command("ric", "start_ric")
        time.sleep(8)
        
        # Start gNB
        print("Starting gNB...")
        self.node_controller._run_command("gnb", "start_gnb")
        time.sleep(8)
        
        # Start UE1 and wait for connection
        print("Starting UE1...")
        self.node_controller._run_command("ue1", "online_mode")
        while True:
            if self.node_controller._run_command("ue1", "check_status"):
                print("UE1 connected")
                break
            time.sleep(0.5)
            
        # Start UE3 and wait for connection
        print("Starting UE3...")
        self.node_controller._run_command("ue3", "online_mode")
        while True:
            if self.node_controller._run_command("ue3", "check_status"):
                print("UE3 connected")
                break
            time.sleep(0.5)
            
        # Start iperf clients
        print("Starting iperf clients...")
        self.node_controller._run_command("ue1", "start_iperf_client")
        self.node_controller._run_command("ue3", "start_iperf_client")
        
        # Start AIML xApp and wait for API
        # import ipdb; ipdb.set_trace()
        print("Starting AIML xApp...")
        self.node_controller._run_command("aiml_xapp", "start_xapp")
        
        # Check if xApp API becomes available
        if not self._check_xapp_api():
            print("Failed to connect to xApp API after maximum retries")
            raise RuntimeError("xApp API not available")
        
        # Initialize PRB ratios
        self.prb_ratios = {1: {"min_prb_ratio": 50, "max_prb_ratio": 100}, 3: {"min_prb_ratio": 0, "max_prb_ratio": 100}}
        self._update_prb_ratio(1, 1, 1, 50, 100)
        self._update_prb_ratio(3, 1, 2, 0, 100)
        print("Setup complete!")

    def run(self):
        """Run the experiment."""
        print("Starting experiment...")
        
        # Set up experiment components
        self.setup_experiment()
            
        try:
            with open(self.data_file, "w") as f:
                for atten in self.config["attens"]:
                    print(f"\nSetting attenuation to {atten}")
                    self._update_atten(atten, ue=1)
                    print(f"Current attenuation values: {self.attens}")

                    # Iterate through each PRB configuration
                    for prb_config in self.config["prb_configs"]:
                        ue_id = prb_config["ue_id"]
                        print(f"Configuring UE {ue_id} with {prb_config}")
                        self._update_prb_ratio(
                            ue_id=ue_id,
                            sst=prb_config["sst"],
                            sd=prb_config["sd"],
                            min_prb_ratio=prb_config["min_prb_ratio"],
                            max_prb_ratio=prb_config["max_prb_ratio"]
                        )
                        
                        # Collect measurements for duration `dwell_time_s`
                        measurements_collected = 0
                        for iteration in range(self.config['dwell_time_s']):
                            try:
                                print(f"\nCollecting measurements (iteration {iteration+1}/{self.config['dwell_time_s']})")
                                response = requests.get(f"{self.base_url}/measurements", timeout=5)
                                response.raise_for_status()
                                measurements = response.json()
                                
                                if not measurements:
                                    print("Warning: Received empty measurements dictionary")
                                    time.sleep(1)
                                    continue
                                
                                print(f"Received measurements for UEs: {list(measurements.keys())}")
                                
                                for ue_id_str, data in measurements.items():
                                    # Start with basic measurement data
                                    ue_id = int(ue_id_str)
                                    
                                    # Skip if UE doesn't have PRB settings
                                    if ue_id not in self.prb_ratios:
                                        print(f"Warning: No PRB settings for UE {ue_id}, skipping")
                                        continue
                                        
                                    # Skip if UE doesn't have attenuation settings
                                    if ue_id not in self.attens:
                                        print(f"Warning: No attenuation settings for UE {ue_id}, skipping")
                                        continue
                                    
                                    measurement_dict = {
                                        "timestamp": datetime.now().isoformat(),
                                        "collet_start_time": data.get("colletStartTime"),
                                        "ue_id": ue_id,
                                        "sst": self.prb_ratios[ue_id].get("sst", 1),
                                        "sd": self.prb_ratios[ue_id].get("sd", 1),
                                        "min_prb_ratio": self.prb_ratios[ue_id]["min_prb_ratio"],
                                        "max_prb_ratio": self.prb_ratios[ue_id]["max_prb_ratio"],
                                        "atten": self.attens[ue_id]
                                    }

                                    # Add measurement data directly to parent dict
                                    meas_data = data.get("measData", {})
                                    if not meas_data:
                                        print(f"Warning: No measurement data for UE {ue_id}")
                                    
                                    for key, value in meas_data.items():
                                        measurement_dict[key] = value[0] if isinstance(value, (list, tuple)) else value

                                    print(f"Writing measurement for UE {ue_id}")
                                    print(f"  Attenuation: {measurement_dict['atten']}")
                                    print(f"  Min PRB Ratio: {measurement_dict['min_prb_ratio']}")
                                    print(f"  Max PRB Ratio: {measurement_dict['max_prb_ratio']}")
                                    
                                    # Write to file
                                    f.write(json.dumps(measurement_dict) + "\n")
                                    f.flush()
                                    
                                    # Store in InfluxDB
                                    self._store_in_influxdb(measurement_dict)
                                    
                                    measurements_collected += 1

                            except RequestException as e:
                                print(f"Error collecting measurements: {e}")
                            except Exception as e:
                                print(f"Unexpected error during measurement collection: {e}")
                                import traceback
                                traceback.print_exc()

                            time.sleep(1)
                            
                        print(f"Collected {measurements_collected} measurements for this configuration")

        except Exception as e:
            print(f"Error in experiment run: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Save docker logs
            print("Saving docker logs...")
            self.node_controller._run_command("ric", "get_docker_logs")
            
            # Stop nodes
            self.node_controller.stop_nodes()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment runner for AIML xApp")
    parser.add_argument("--config", type=str, default="default", help="Experiment configuration")
    parser.add_argument("--experiment_id", type=str, help="Experiment ID")
    parser.add_argument("--run_id", type=str, help="Run ID")
    return parser.parse_args()

def main():
    """Main entry point for the experiment runner."""
    args = parse_arguments()
    if args.config == "default":
        config = DEFAULT_EXP_CONFIG
    elif args.config == "test":
        config = TEST_EXP_CONFIG
    else:
        raise ValueError(f"Invalid configuration: {args.config}")
    runner = ExperimentRunner(
        config=config, 
        experiment_id=args.experiment_id,
        run_id=args.run_id
    )
    runner.run()

if __name__ == "__main__":
    main()
