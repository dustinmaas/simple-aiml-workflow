#!/bin/bash
# InfluxDB Network Metrics Backup Script
# This script automatically starts the InfluxDB container if needed,
# performs a backup of the network_metrics bucket, and stops the container
# if it wasn't already running.

set -e

# Default values
BACKUP_DIR=""
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONTAINER_MOUNT_DIR="/backups"
CONTAINER_NAME="datalake_influxdb"
BUCKET="network_metrics"
ORG="ric"

# Help message
function show_help {
    echo "Usage: $0 [OPTIONS]"
    echo "Backup the InfluxDB network_metrics bucket"
    echo
    echo "Options:"
    echo "  -d, --backup-dir DIR   Backup directory (default: ./datalake/backups/YYYYMMDD_HHMMSS)"
    echo "  -b, --bucket BUCKET    Bucket name (default: network_metrics)"
    echo "  -o, --org ORG          Organization name (default: ric)"
    echo "  -h, --help             Display this help message and exit"
    echo
    echo "Example:"
    echo "  $0 --backup-dir ./datalake/backups/custom_backup"
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
    BACKUP_DIR="./datalake/backups/$TIMESTAMP"
fi

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"
BACKUP_DIR=$(realpath "$BACKUP_DIR")
echo "Backup will be stored in: $BACKUP_DIR"

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

# The backup script is already mounted in the container at /scripts/influxdb_backup.sh
echo "Using mounted backup script in the container..."
sudo docker exec datalake_influxdb chmod +x "/scripts/influxdb_backup.sh"

# Use the mounted backups directory in the container
CONTAINER_BACKUP_DIR="$CONTAINER_MOUNT_DIR/$(basename "$BACKUP_DIR")"
sudo docker exec datalake_influxdb mkdir -p "$CONTAINER_BACKUP_DIR"

# Run backup inside container
echo "Starting backup process..."
sudo docker exec datalake_influxdb /scripts/influxdb_backup.sh \
    --backup-dir "$CONTAINER_MOUNT_DIR/$(basename "$BACKUP_DIR")" \
    --bucket "$BUCKET" \
    --org "$ORG"

# No need to copy - the files are already in the mounted volume directory
echo "Backup files are available in $BACKUP_DIR"

# Shut down the container if it wasn't running before
if [ "$CONTAINER_RUNNING" = false ]; then
    echo "Shutting down the service since it was not running before..."
    # Use docker compose down to completely remove the container and networks
    # instead of just stopping the container
    sudo docker compose down
fi

# Create/update symlink to latest backup
LATEST_SYMLINK="./datalake/backups/latest"
ln -sf "$BACKUP_DIR" "$LATEST_SYMLINK"
echo "Updated symlink: $LATEST_SYMLINK -> $BACKUP_DIR"

echo "Backup completed successfully and stored in: $BACKUP_DIR"
echo "To restore this backup, use: ./datalake/restore_influxdb_metrics.sh --backup-dir $BACKUP_DIR"
echo "Or simply use: ./datalake/restore_influxdb_metrics.sh (will use latest backup)"
