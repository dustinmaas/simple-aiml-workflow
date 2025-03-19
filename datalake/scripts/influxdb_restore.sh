#!/bin/bash
# InfluxDB Restore Script
# This script restores the InfluxDB network_metrics bucket from a backup
# It should be executed inside the InfluxDB container

set -e

# Default values
BACKUP_DIR="/backups/latest"
BUCKET="${DOCKER_INFLUXDB_INIT_BUCKET:-network_metrics}"
ORG="${DOCKER_INFLUXDB_INIT_ORG:-ric}"
TOKEN="${DOCKER_INFLUXDB_INIT_ADMIN_TOKEN:-ric_admin_token}"

# Help message
function show_help {
    echo "Usage: $0 [OPTIONS]"
    echo "Restore the InfluxDB network_metrics bucket from a backup"
    echo
    echo "Options:"
    echo "  -d, --backup-dir DIR   Backup directory (default: /backups/latest)"
    echo "  -b, --bucket BUCKET    Bucket name (default: network_metrics)"
    echo "  -o, --org ORG          Organization name (default: ric)"
    echo "  -t, --token TOKEN      Admin token (default: from environment)"
    echo "  -h, --help             Display this help message and exit"
    echo
    echo "Example:"
    echo "  $0 --backup-dir /backups/20250318_123456"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -d|--backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        -b|--bucket)
            BUCKET="$2"
            shift 2
            ;;
        -o|--org)
            ORG="$2"
            shift 2
            ;;
        -t|--token)
            TOKEN="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if backup directory exists
if [ ! -d "$BACKUP_DIR" ]; then
    echo "ERROR: Backup directory does not exist: $BACKUP_DIR"
    exit 1
fi

# Check for backup metadata file
if [ -f "$BACKUP_DIR/backup_metadata.txt" ]; then
    echo "Found backup metadata:"
    cat "$BACKUP_DIR/backup_metadata.txt"
    echo ""
fi

# Check if bucket already exists
echo "Checking if bucket '$BUCKET' already exists..."
BUCKET_ID=$(influx bucket list -t "$TOKEN" -o "$ORG" --name "$BUCKET" --hide-headers | awk '{print $1}')

if [ -n "$BUCKET_ID" ]; then
    echo "WARNING: Bucket '$BUCKET' already exists (ID: $BUCKET_ID)"
    echo "Restoring will overwrite all existing data in this bucket."
    echo ""
    
    # Ask for confirmation
    read -p "Delete existing bucket and continue with restore? (y/N): " CONFIRM
    
    if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "Deleting existing bucket..."
        influx bucket delete -t "$TOKEN" -i "$BUCKET_ID"
        echo "Bucket deleted successfully."
    else
        echo "Restore cancelled by user."
        exit 0
    fi
else
    echo "Bucket '$BUCKET' does not exist. It will be created during restore."
fi

# Run the restore command
echo "Starting restore of bucket '$BUCKET' from '$BACKUP_DIR'..."

# Perform the restore operation
influx restore "$BACKUP_DIR" \
    --bucket "$BUCKET" \
    --org "$ORG" \
    --token "$TOKEN"

# Check if restore was successful
if [ $? -eq 0 ]; then
    echo "Restore completed successfully from: $BACKUP_DIR"
    echo "Restored to bucket: $BUCKET"
    echo "Restore process complete."
else
    echo "Restore failed!"
    exit 1
fi
