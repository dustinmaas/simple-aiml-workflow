#!/usr/bin/env python3
"""
AIML xApp with REST API for OSC RIC.

This module implements an xApp that provides measurement collection and PRB ratio control
through a REST API interface.
"""

# Standard library imports
import json
import logging
import time
import datetime
import argparse
import signal
import threading

# Third-party imports
from flask import Flask, request, jsonify

# Local imports
from lib.xAppBase import xAppBase

# Constants
DEFAULT_HTTP_PORT = 8090
DEFAULT_RMR_PORT = 4560
DEFAULT_E2_NODE = "gnbd_999_099_0000019b_0"
DEFAULT_RAN_FUNC_ID = 2
DEFAULT_KPM_STYLE = 2
DEFAULT_METRICS = (
    "CQI,RSRP,RSRQ,DRB.UEThpDl,DRB.UEThpUl,"
    "DRB.RlcPacketDropRateDl,DRB.PacketSuccessRateUlgNBUu,"
    "DRB.RlcSduTransmittedVolumeDL,DRB.RlcSduTransmittedVolumeUL"
)
DEFAULT_MIN_PRB_RATIO = 0
DEFAULT_MAX_PRB_RATIO = 100
DEFAULT_SST = 1
DEFAULT_SD = 1
DEFAULT_UE_ID = 1
VALID_SD_VALUES = [1, 2]


class MyXapp(xAppBase):
    def __init__(self, http_server_port, rmr_port):
        super().__init__("", http_server_port, rmr_port)
        self.latest_measurements = {}

    def my_subscription_callback(
        self,
        e2_agent_id,
        subscription_id,
        indication_hdr,
        indication_msg,
        kpm_report_style,
        ue_id,
    ):
        indication_hdr = self.e2sm_kpm.extract_hdr_info(indication_hdr)
        meas_data = self.e2sm_kpm.extract_meas_data(indication_msg)
        meas_data["colletStartTime"] = indication_hdr["colletStartTime"]
        meas_data["ue_id"] = ue_id
        self.latest_measurements[ue_id] = meas_data
        print(f"Received measurement for UE {ue_id}: {meas_data}")

    @xAppBase.start_function
    def start(self, e2_node_id, kpm_report_style, ue_ids, metric_names):
        report_period = 1000
        granul_period = 1000

        if kpm_report_style == 2:
            subscription_callback0 = lambda agent, sub, hdr, msg: self.my_subscription_callback(
                agent, sub, hdr, msg, kpm_report_style, ue_ids[0]
            )
            subscription_callback1 = lambda agent, sub, hdr, msg: self.my_subscription_callback(
                agent, sub, hdr, msg, kpm_report_style, ue_ids[1]
            )
            print(
                "Subscribe to E2 node ID: {}, RAN func: e2sm_kpm, Report Style: {}, UE_id: {}, metrics: {}".format(
                    e2_node_id, kpm_report_style, ue_ids[0], metric_names
                )
            )
            self.e2sm_kpm.subscribe_report_service_style_2(
                e2_node_id,
                report_period,
                ue_ids[0],
                metric_names,
                granul_period,
                subscription_callback0,
            )
            print(
                "Subscribe to E2 node ID: {}, RAN func: e2sm_kpm, Report Style: {}, UE_id: {}, metrics: {}".format(
                    e2_node_id, kpm_report_style, ue_ids[1], metric_names
                )
            )
            self.e2sm_kpm.subscribe_report_service_style_2(
                e2_node_id,
                report_period,
                ue_ids[1],
                metric_names,
                granul_period,
                subscription_callback1,
            )
        else:
            print(
                "INFO: Subscription for E2SM_KPM Report Service Style {} is not supported".format(
                    kpm_report_style
                )
            )
            exit(1)


class AIMLXAppServer:
    """Main server class that encapsulates the xApp and API server."""
    
    def __init__(self):
        self.exp_api_server = Flask(__name__)
        # Set Flask app log level to INFO
        self.exp_api_server.logger.setLevel(logging.INFO)
        # Ensure all handlers use INFO level
        for handler in self.exp_api_server.logger.handlers:
            handler.setLevel(logging.INFO)
        self.aiml_xapp = None
        self.ue_ids = []
        self._setup_routes()
        
    def _setup_routes(self):
        """Set up Flask routes."""
        self.exp_api_server.before_request(self.log_request_info)
        self.exp_api_server.route("/measurements", methods=["GET"])(self.get_measurements)
        self.exp_api_server.route("/update_prb_ratio", methods=["POST"])(self.update_prb_ratio)
        self.exp_api_server.route("/health", methods=["GET"])(self.health_check)

    def log_request_info(self):
        """Log request information."""
        self.exp_api_server.logger.info("Headers: %s", dict(request.headers))
        self.exp_api_server.logger.info("Body: %s", request.get_data())

    def get_measurements(self):
        """Handle GET /measurements endpoint."""
        ue_id = request.args.get("ue_id")
        self.exp_api_server.logger.info("\nGET /measurements")
        self.exp_api_server.logger.info(f"Query params: {request.args}")

        if ue_id is not None:
            try:
                ue_id = str(int(ue_id))  # Validate it's a number and convert to string
                if ue_id in self.aiml_xapp.latest_measurements:
                    response = {ue_id: self.aiml_xapp.latest_measurements[ue_id]}
                    json_response = json.dumps(response, cls=DateTimeEncoder)
                    self.exp_api_server.logger.info(f"Response: {json_response}")
                    return json_response, 200, {"Content-Type": "application/json"}
                else:
                    self.exp_api_server.logger.error(f"No measurements found for UE {ue_id}")
                    return jsonify({"error": f"No measurements found for UE {ue_id}"}), 404
            except ValueError:
                self.exp_api_server.logger.error("ue_id must be an integer")
                return jsonify({"error": "ue_id must be an integer"}), 400

        response = self.aiml_xapp.latest_measurements
        json_response = json.dumps(response, cls=DateTimeEncoder)
        self.exp_api_server.logger.info(f"Response: {json_response}")
        return json_response, 200, {"Content-Type": "application/json"}

    def update_prb_ratio(self):
        """Handle POST /update_prb_ratio endpoint."""
        try:
            data = request.get_json()
        except Exception as e:
            error = {"error": "Invalid JSON in request body"}
            self.exp_api_server.logger.error(f"JSON parsing error: {str(e)}")
            return jsonify(error), 400

        self.exp_api_server.logger.info("\nPOST /update_prb_ratio")
        self.exp_api_server.logger.info(f"Request body: {json.dumps(data, cls=DateTimeEncoder)}")

        try:
            ue_id = int(data.get("ue_id", DEFAULT_UE_ID))
            sst = int(data.get("sst", DEFAULT_SST))
            sd = int(data.get("sd", DEFAULT_SD))
            min_prb_ratio = int(data.get("min_prb_ratio", DEFAULT_MIN_PRB_RATIO))
            max_prb_ratio = int(data.get("max_prb_ratio", DEFAULT_MAX_PRB_RATIO))

            if sd not in VALID_SD_VALUES:
                error = {"error": f"sd must be in {VALID_SD_VALUES}"}
                self.exp_api_server.logger.error(f"Validation error: {error}")
                return jsonify(error), 400

            if ue_id not in self.ue_ids:
                error = {"error": f"ue_id must be in {self.ue_ids}"}
                self.exp_api_server.logger.error(f"Validation error: {error}")
                return jsonify(error), 400

            self.aiml_xapp.e2sm_rc.control_slice_level_prb_quota(
                "gnbd_999_099_0000019b_0",
                ue_id - 1,
                sst,
                sd,
                min_prb_ratio=min_prb_ratio,
                max_prb_ratio=max_prb_ratio,
                dedicated_prb_ratio=100,
                ack_request=1,
            )

            response = {
                "message": f"PRB ratio set to min {min_prb_ratio} max {max_prb_ratio} for UE {ue_id} (slice {sd}, sst {sst})"
            }
            self.exp_api_server.logger.info(f"Response: {json.dumps(response)}")
            return jsonify(response)

        except Exception as e:
            error = {"error": f"Failed to set PRB ratio: {str(e)}"}
            self.exp_api_server.logger.error(f"E2SM-RC error: {str(e)}")
            return jsonify(error), 500

    def health_check(self):
        """Handle GET /health endpoint."""
        if self.aiml_xapp and self.aiml_xapp.running:
            return jsonify({"status": "healthy"}), 200
        return jsonify({"status": "unhealthy"}), 503

    def setup(self, args):
        """Set up the server with command line arguments."""
        self.ue_ids = list(map(int, args.ue_ids.split(",")))
        print(f"UE IDs: {self.ue_ids}")
        self.aiml_xapp = MyXapp(args.http_server_port, args.rmr_port)
        self.aiml_xapp.e2sm_kpm.set_ran_func_id(args.ran_func_id)
        setup_signal_handlers(self.aiml_xapp)
        thread = threading.Thread(
            target=lambda: self.aiml_xapp.start(
                args.e2_node_id, args.kpm_report_style, self.ue_ids, args.metrics.split(",")
            )
        )
        thread.start()
        return thread

    def run(self, host="0.0.0.0", port=61611):
        """Run the server."""
        self.exp_api_server.run(host=host, port=port)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)


def validate_prb_params(data):
    try:
        ue_id = int(data.get("ue_id", DEFAULT_UE_ID))
        sst = int(data.get("sst", DEFAULT_SST))
        sd = int(data.get("sd", DEFAULT_SD))
        min_prb_ratio = int(data.get("min_prb_ratio", DEFAULT_MIN_PRB_RATIO))
        max_prb_ratio = int(data.get("max_prb_ratio", DEFAULT_MAX_PRB_RATIO))
    except (TypeError, ValueError):
        raise ValueError("Parameters must be integers")

    if sd not in VALID_SD_VALUES:
        raise ValueError(f"sd must be in {VALID_SD_VALUES}")

    if ue_id not in ue_ids:
        raise ValueError(f"ue_id must be in {ue_ids}")

    return min_prb_ratio, max_prb_ratio, sd, sst, ue_id


def parse_arguments():
    parser = argparse.ArgumentParser(description="AIML xApp for OSC RIC")
    parser.add_argument(
        "--http_server_port",
        type=int,
        default=DEFAULT_HTTP_PORT,
        help="HTTP server listen port",
    )
    parser.add_argument(
        "--rmr_port", type=int, default=DEFAULT_RMR_PORT, help="RMR port"
    )
    parser.add_argument(
        "--e2_node_id", type=str, default=DEFAULT_E2_NODE, help="E2 Node ID"
    )
    parser.add_argument(
        "--ran_func_id",
        type=int,
        default=DEFAULT_RAN_FUNC_ID,
        help="RAN function ID",
    )
    parser.add_argument(
        "--kpm_report_style",
        type=int,
        default=DEFAULT_KPM_STYLE,
        help="KPM Report Style ID",
    )
    parser.add_argument("--ue_ids", type=str, default=DEFAULT_UE_ID, help="UE ID")
    parser.add_argument(
        "--metrics",
        type=str,
        default=DEFAULT_METRICS,
        help="Metrics name as comma-separated string",
    )
    return parser.parse_args()


def setup_signal_handlers(xapp):
    for sig in (signal.SIGQUIT, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, xapp.signal_handler)



def main():
    args = parse_arguments()
    server = AIMLXAppServer()
    xapp_thread = server.setup(args)

    try:
        server.run()
    finally:
        if server.aiml_xapp:
            server.aiml_xapp.running = False
            server.aiml_xapp.signal_handler(None, None)
        if xapp_thread.is_alive():
            xapp_thread.join()


if __name__ == "__main__":
    main()
