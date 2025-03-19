#!/bin/bash
# InfluxDB Network Metrics Restore Script
# This script automatically starts the InfluxDB container if needed,
# restores the network_metrics bucket from a backup, and stops the container
# if it wasn't already running.

set -e

# Default values
BACKUP_DIR=""
CONTAINER_MOUNT_DIR="/backups"
CONTAINER_NAME="datalake_influxdb"
BUCKET="network_metrics"
ORG="ric"

# Help message
function show_help {
    echo "Usage: $0 [OPTIONS]"
    echo "Restore the InfluxDB network_metrics bucket from a backup"
    echo
    echo "Options:"
    echo "  -d, --backup-dir DIR   Backup directory (default: ./datalake/backups/latest)"
    echo "  -b, --bucket BUCKET    Bucket name (default: network_metrics)"
    echo "  -o, --org ORG          Organization name (default: ric)"
    echo "  -h, --help             Display this help message and exit"
    echo
    echo "Example:"
    echo "  $0 --backup-dir ./datalake/backups/20250318_123456"
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

# Set default backup directory if not specified
if [ -z "$BACKUP_DIR" ]; then
    BACKUP_DIR="./datalake/backups/latest"
    # Create symlink to most recent backup if it doesn't exist
    if [ ! -d "$BACKUP_DIR" ] && [ -d "./datalake/backups" ]; then
        LATEST=$(ls -td ./datalake/backups/202* | head -1)
        if [ -n "$LATEST" ]; then
            ln -sf "$LATEST" "$BACKUP_DIR"
            echo "Using most recent backup: $LATEST"
        fi
    fi
fi

# Check if backup directory exists
if [ ! -d "$BACKUP_DIR" ]; then
    echo "ERROR: Backup directory does not exist: $BACKUP_DIR"
    exit 1
fi

BACKUP_DIR=$(realpath "$BACKUP_DIR")
echo "Using backup from: $BACKUP_DIR"

# Check if docker compose is working
if ! sudo docker compose version &> /dev/null; then
    echo "ERROR: docker compose is not installed or not working"
    exit 1
fi

# Check if container is already running
CONTAINER_RUNNING=false
if sudo docker ps | grep -q datalake_influxdb; then
    CONTAINER_RUNNING=true
    echo "InfluxDB container is already running."
else
    echo "InfluxDB container is not running, starting it now..."
    # Start only the InfluxDB service
    sudo docker compose up -d influxdb
    
    # Wait for container to be ready
    echo "Waiting for InfluxDB to be ready..."
    for i in {1..30}; do
        if sudo docker compose exec influxdb curl -s http://localhost:8086/health &> /dev/null; then
            echo "InfluxDB is ready."
            break
        fi
        
        if [ $i -eq 30 ]; then
            echo "ERROR: Timed out waiting for InfluxDB to be ready"
            if [ "$CONTAINER_RUNNING" = false ]; then
                echo "Shutting down the service since it was not running before..."
                sudo docker compose down
            fi
            exit 1
        fi
        
        echo "Waiting for InfluxDB to be ready... ($i/30)"
        sleep 2
    done
fi

# The restore script is already mounted in the container at /scripts/influxdb_restore.sh
echo "Using mounted restore script in the container..."
sudo docker exec datalake_influxdb chmod +x "/scripts/influxdb_restore.sh"

# If the backup is already in the mounted backups directory, use it directly
BACKUP_BASE=$(basename "$BACKUP_DIR")
BACKUP_PARENT=$(dirname "$BACKUP_DIR")
BACKUP_PARENT_BASE=$(basename "$BACKUP_PARENT")

if [ "$BACKUP_PARENT_BASE" = "backups" ] && [ -d "$BACKUP_DIR" ]; then
    # The backup is in the mounted backups directory, use it directly
    CONTAINER_BACKUP_DIR="$CONTAINER_MOUNT_DIR/$BACKUP_BASE"
    echo "Using backup directly from mounted directory: $CONTAINER_BACKUP_DIR"
else
    # The backup is elsewhere, copy it to the container
    CONTAINER_BACKUP_DIR="/tmp/influxdb_restore_$(date +%s)"
    sudo docker exec datalake_influxdb mkdir -p "$CONTAINER_BACKUP_DIR"
    
    echo "Copying backup data to container..."
    sudo docker cp "$BACKUP_DIR/." "datalake_influxdb:$CONTAINER_BACKUP_DIR"
fi

# Display backup metadata if it exists
if [ -f "$BACKUP_DIR/backup_metadata.txt" ]; then
    echo "Backup metadata:"
    cat "$BACKUP_DIR/backup_metadata.txt"
    echo ""
fi

# Run restore inside container
echo "Starting restore process..."
echo "The script will check if the bucket exists and prompt for confirmation if needed"

sudo docker exec -it datalake_influxdb /scripts/influxdb_restore.sh \
    --backup-dir "$CONTAINER_BACKUP_DIR" \
    --bucket "$BUCKET" \
    --org "$ORG"

# Clean up temporary backup directory if we created one
if [[ "$CONTAINER_BACKUP_DIR" == /tmp/* ]]; then
    echo "Cleaning up temporary files..."
    sudo docker exec datalake_influxdb rm -rf "$CONTAINER_BACKUP_DIR"
fi

# Shut down the container if it wasn't running before
if [ "$CONTAINER_RUNNING" = false ]; then
    echo "Shutting down the service since it was not running before..."
    # Use docker compose down to completely remove the container and networks
    # instead of just stopping the container
    sudo docker compose down
fi

echo "Restore completed successfully from: $BACKUP_DIR"
