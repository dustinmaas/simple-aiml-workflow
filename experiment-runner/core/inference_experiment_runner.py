#!/usr/bin/env python3
"""
Inference experiment runner module.

Extends the base experiment runner to support ML-based inference
for PRB ratio prediction based on CQI and attenuation values.
"""

import os
import json
import time
import sys
import logging

from typing import Dict, Any, Optional

import numpy as np
import onnxruntime as ort

from huggingface_hub import hf_hub_download
from influxdb_client.client.write_api import SYNCHRONOUS

from core.experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)

class InferenceExperimentRunner(ExperimentRunner):
    """
    Runner for inference experiments.
    
    Uses ML models to predict min_prb_ratio values based on CQI and target
    downlink throughput.
    """
    
    def __init__(
        self,
        node_controller,
        config: Dict[str, Any],
        model_repo: str,
        model_file: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize inference experiment runner.
        
        Args:
            node_controller: NodeController instance for remote command execution
            config: Experiment configuration
            model_repo: HuggingFace repository containing the model
            model_file: Name of the model file in the repository (if None, will auto-detect)
            **kwargs: Additional arguments to pass to the parent class
        """
        super().__init__(node_controller, config, **kwargs)
        
        self.model_repo = model_repo
        self.model_file = model_file
        
        # Initialize state variables for prediction
        self.model = None
        self.prediction_metrics = {}
        self.run_id = str(int(time.time()))  # Generate a timestamp run ID for this experiment
        
        # Load the model
        self._load_model()
        
    def _ensure_influxdb_bucket(self) -> None:
        """
        Ensure the InfluxDB bucket for student model metrics exists.
        Creates the bucket if it doesn't exist.
        """
        if not hasattr(self, 'influxdb_client') or not self.influxdb_client:
            logger.warning("InfluxDB client not available. Required environment variables may not be set.")
            return
            
        try:
            buckets_api = self.influxdb_client.buckets_api()
            bucket_name = "student_model_evals"
            
            # Get list of existing buckets
            buckets = buckets_api.find_buckets().buckets
            bucket_names = [bucket.name for bucket in buckets]
            
            # Create bucket if it doesn't exist
            if bucket_name not in bucket_names:
                logger.info(f"Creating InfluxDB bucket '{bucket_name}'")
                buckets_api.create_bucket(bucket_name=bucket_name, org=self.influxdb_org)
                logger.info(f"Successfully created InfluxDB bucket '{bucket_name}'")
            else:
                logger.info(f"InfluxDB bucket '{bucket_name}' already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring InfluxDB bucket exists: {e}")
    
    def _load_model(self) -> None:
        """
        Load the ML model from HuggingFace.
        
        If model_file is not specified, this will automatically detect
        the first ONNX model in the repository.
        """
        try:
            logger.info(f"Loading model from repository {self.model_repo}")
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            
            # If model_file not specified, find the first ONNX model
            if not self.model_file:
                from huggingface_hub import list_repo_files
                
                try:
                    repo_files = list_repo_files(repo_id=self.model_repo, token=hf_token)
                    onnx_files = [f for f in repo_files if f.endswith('.onnx')]
                    
                    if not onnx_files:
                        raise ValueError(f"No ONNX models found in repository {self.model_repo}")
                    
                    self.model_file = onnx_files[0]
                    logger.info(f"Automatically selected ONNX model: {self.model_file}")
                except Exception as e:
                    logger.error(f"Error finding ONNX model in repository: {e}")
                    raise
            
            # Download the model file
            logger.info(f"Downloading model file: {self.model_file}")
            model_path = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.model_file,
                token=hf_token
            )
            
            # Load the model
            self.model = ort.InferenceSession(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
            
            # Verify the model with test input format [[15, 200]]
            self._verify_model_input_format()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _verify_model_input_format(self) -> None:
        """
        Verify that the model accepts input in the format [[15, 200]].
        If verification fails, exit.
        """
        try:
            if not self.model:
                logger.error("Cannot verify model: Model not loaded")
                sys.exit(1)
                
            # Create test input with the required format [[15, 200]]
            test_input = np.array([[15.0, 200.0]], dtype=np.float32)
            input_data = {"input": test_input}
            
            logger.info("Verifying model with test input [[15, 200]]")
            
            try:
                # Run prediction with test input
                outputs = self.model.run(None, input_data)
                
                # Verify that outputs is not empty and has the expected format
                if not outputs or not isinstance(outputs, list) or len(outputs) == 0:
                    logger.error("Model verification failed: Invalid output format (empty or not a list)")
                    sys.exit(1)
                    
                # Check that the first output is a numpy array with at least one value
                if not isinstance(outputs[0], np.ndarray) or outputs[0].size == 0:
                    logger.error("Model verification failed: Output is not a valid numpy array")
                    sys.exit(1)
                    
                logger.info(f"Model verification successful. Output shape: {outputs[0].shape}, type: {type(outputs[0])}")
                
            except Exception as e:
                logger.error(f"Model verification failed: {e}")
                logger.error("The model does not accept input in the format [[15, 200]]")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Error during model verification: {e}")
            sys.exit(1)
    
    def _get_cqi_from_measurements(self, ue_id: int) -> Optional[int]:
        """
        Extract CQI value from measurements for a specific UE.
        
        Args:
            ue_id: UE identifier
            
        Returns:
            CQI value (as integer) or None if not available
        """
        try:
            measurements = self._collect_measurements()
            logger.debug(f"Measurements collected: {measurements}")
            if not measurements:
                raise ValueError("Empty measurements dictionary received")
            if str(ue_id) not in measurements:
                raise ValueError(f"No measurements available for UE {ue_id}")
                
            ue_data = measurements[str(ue_id)]
            meas_data = ue_data.get("measData", {})
            
            cqi = meas_data.get("CQI")
            if cqi is None:
                logger.warning(f"No CQI data available for UE {ue_id}")
                return None
                
            # If CQI is a list, take the first value
            if isinstance(cqi, list):
                cqi = cqi[0]
                
            return int(cqi)
            
        except Exception as e:
            logger.error(f"Error getting CQI for UE {ue_id}: {e}")
            return None
    
    def _predict_min_prb_ratio(self, target_thp: float, cqi: int) -> int:
        """
        Predict the minimum PRB ratio needed to achieve the target throughput.
        
        Args:
            target_thp: Target throughput (float)
            cqi: Channel Quality Indicator (integer)
            
        Returns:
            Predicted minimum PRB ratio (clipped to [50,95])
        """
        try:
            if not self.model:
                logger.error("Model not loaded")
                return 50  # Default value if model not available
                
            # Prepare input for the model - only using CQI and target_thp
            input_data = {
                "input": np.array([[float(cqi), float(target_thp)]], dtype=np.float32)
            }
            
            # Run prediction
            logger.info(f"Predicting min_prb_ratio for target_thp={target_thp}, cqi={cqi}")
            outputs = self.model.run(None, input_data)
            
            # Extract the prediction (assuming single output)
            logger.debug(f"Model outputs: {outputs}")
            prediction = float(outputs[0][0])
            
            # Ensure the result is within bounds (50-95)
            min_prb_ratio = max(50, min(95, int(round(prediction))))
            
            logger.info(f"Predicted min_prb_ratio: {min_prb_ratio}")
            return min_prb_ratio
            
        except Exception as e:
            logger.error(f"Error predicting min_prb_ratio: {e}")
            return 50  # Default value if prediction fails
    
    def _store_student_model_metrics(
        self,
        ue_id: int,
        atten: int,
        target_thp: float,
        predicted_min_prb: int,
        actual_thp: Optional[float] = None,
        cqi: Optional[int] = None
    ) -> None:
        """
        Store student model metrics in InfluxDB.
        
        Args:
            ue_id: UE identifier
            atten: Attenuation value
            target_thp: Target throughput
            predicted_min_prb: Predicted minimum PRB ratio
            actual_thp: Actual throughput achieved (if available)
            cqi: Channel Quality Indicator used for prediction
        """
        if not hasattr(self, 'influxdb_write_api') or not self.influxdb_write_api:
            return
            
        try:
            # Fields are the numeric metrics
            fields = {
                "target_thp": float(target_thp),
                "predicted_min_prb": int(predicted_min_prb)
            }
            
            # Add optional fields
            if actual_thp is not None:
                fields["actual_thp"] = float(actual_thp)
            if cqi is not None:
                fields["cqi"] = int(cqi)
            
            # Store in InfluxDB
            self.influxdb_write_api.write(
                bucket="student_model_evals", 
                record=[{
                    "measurement": "student_model_eval",
                    "tags": {
                        "run_id": self.run_id,
                        "model_repo": self.model_repo,
                        "ue_id": str(ue_id),
                        "atten": str(atten)
                    },
                    "fields": fields
                }]
            )
        except Exception as e:
            logger.error(f"Error storing metrics in InfluxDB: {e}")
            
    def _store_prediction_metrics(
        self,
        ue_id: int,
        atten: int,
        target_thp: float,
        predicted_min_prb: int,
        actual_thp: Optional[float] = None,
        cqi: Optional[int] = None
    ) -> None:
        """
        Store prediction metrics for later analysis.
        
        Args:
            ue_id: UE identifier
            atten: Attenuation value
            target_thp: Target throughput
            predicted_min_prb: Predicted minimum PRB ratio
            actual_thp: Actual throughput achieved (if available)
            cqi: Channel Quality Indicator used for prediction
        """
        key = f"{ue_id}_{atten}_{target_thp}"
        
        self.prediction_metrics[key] = {
            "ue_id": ue_id,
            "atten": atten,
            "target_thp": target_thp,
            "predicted_min_prb": predicted_min_prb,
            "actual_thp": actual_thp,
            "cqi": cqi,
            "timestamp": time.time()
        }
        
        # Store student model metrics in InfluxDB
        self._store_student_model_metrics(
            ue_id=ue_id,
            atten=atten,
            target_thp=target_thp,
            predicted_min_prb=predicted_min_prb,
            actual_thp=actual_thp,
            cqi=cqi
        )
        
        # Log the metrics
        info_str = f"Prediction metrics - UE: {ue_id}, Atten: {atten}, Target: {target_thp}, Predicted min_prb: {predicted_min_prb}"
        if actual_thp:
            thp_diff = actual_thp - target_thp
            percent_diff = (thp_diff / target_thp) * 100 if target_thp else 0
            info_str += f", Actual thp: {actual_thp} ({percent_diff:.2f}% diff)"
        
        logger.info(info_str)
    
    def _get_actual_throughput(self, ue_id: int) -> Optional[int]:
        """
        Get the actual throughput achieved for a UE.
        
        Args:
            ue_id: UE identifier
            
        Returns:
            Actual throughput or None if not available
        """
        try:
            measurements = self._collect_measurements()
            if not measurements:
                raise ValueError("Empty measurements dictionary received")
            if str(ue_id) not in measurements:
                raise ValueError(f"No measurements available for UE {ue_id}")
                
            ue_data = measurements[str(ue_id)]
            meas_data = ue_data.get("measData", {})
            
            thp = meas_data.get("DRB.UEThpDl")
            if thp is None:
                logger.warning(f"No throughput data available for UE {ue_id}")
                return None
                
            # If throughput is a list, take the first value
            if isinstance(thp, list):
                thp = thp[0]
                
            return int(thp)
            
        except Exception as e:
            logger.error(f"Error getting actual throughput for UE {ue_id}: {e}")
            return None
    
    def _export_prediction_metrics(self) -> None:
        """
        Export prediction metrics to a file.
        """
        if not self.prediction_metrics:
            logger.warning("No prediction metrics to export")
            return
            
        try:
            metrics_file = f"prediction_metrics_{self.experiment_id}.json"
            with open(metrics_file, "w") as f:
                json.dump(self.prediction_metrics, f, indent=2)
                
            logger.info(f"Exported prediction metrics to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error exporting prediction metrics: {e}")
    
    def run(self) -> bool:
        """
        Run the inference experiment.
        
        Returns:
            bool: True if experiment succeeded, False otherwise
        """
        logger.info(f"Starting inference experiment (run_id: {self.run_id}, model_repo: {self.model_repo})...")
        self.running = True
        self.prediction_metrics = {}
        
        try:
            # Set up experiment components
            if not self._setup_experiment():
                logger.error("Failed to set up experiment")
                return False
                
            # Ensure InfluxDB bucket exists
            self._ensure_influxdb_bucket()
        
            # Create data file
            with open(self.data_file, "w") as f:
                # Empty file
                pass
            
            # Check if we have the required configuration
            if "attens_and_target_thps" not in self.config:
                logger.error("Missing 'attens_and_target_thps' in configuration")
                return False
                
            attens_and_target_thps = self.config["attens_and_target_thps"]
            
            # Default PRB configuration
            default_prb_config = {
                "ue_id": 1,
                "sst": 1,
                "sd": 1,
                "max_prb_ratio": 100  # We'll calculate min_prb_ratio based on prediction
            }
            
            # Run experiment
            for atten, targets in attens_and_target_thps.items():
                atten = int(atten)
                logger.info(f"Setting attenuation to {atten}")
                self._update_atten(atten, ue=1)
                
                # Wait a bit for CQI to stabilize
                time.sleep(5)
                
                # Get current CQI
                cqi = self._get_cqi_from_measurements(1)
                if cqi is None:
                    logger.warning(f"Could not get CQI for attenuation {atten}, using default value")
                    cqi = 15  # Default CQI value (integer) if not available
                
                # For each target throughput in this attenuation setting
                for target_thp in targets:
                    # Convert target_thp to float for prediction
                    target_thp_float = float(target_thp)
                    
                    # Predict min_prb_ratio needed to achieve target throughput
                    min_prb_ratio = self._predict_min_prb_ratio(target_thp_float, cqi)
                    min_prb_ratio = max(50, min(min_prb_ratio, 95))  # Ensure within bounds
                    
                    # Configure UE with predicted min_prb_ratio
                    prb_config = dict(default_prb_config)
                    prb_config["min_prb_ratio"] = min_prb_ratio
                    
                    logger.info(f"Configuring UE 1 with min_prb_ratio={min_prb_ratio} for target_thp={target_thp}")
                    self._update_prb_ratio(
                        ue_id=1,
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
                    
                    # Get latest throughput from measurements
                    actual_thp = self._get_actual_throughput(1)
                    
                    # Store prediction metrics
                    self._store_prediction_metrics(
                        ue_id=1,
                        atten=atten,
                        target_thp=target_thp_float,
                        predicted_min_prb=min_prb_ratio,
                        actual_thp=actual_thp,
                        cqi=cqi
                    )
                    
                    logger.info(f"Collected {measurements_collected} measurements for target thp={target_thp}")

            # Export prediction metrics
            self._export_prediction_metrics()
            
            logger.info("Inference experiment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in inference experiment run: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            self.running = False
            self._cleanup()
