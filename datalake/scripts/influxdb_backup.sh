#!/bin/bash
# InfluxDB Backup Script
# This script backs up the InfluxDB network_metrics bucket
# It should be executed inside the InfluxDB container

set -e

# Default values
BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
BUCKET="${DOCKER_INFLUXDB_INIT_BUCKET:-network_metrics}"
ORG="${DOCKER_INFLUXDB_INIT_ORG:-ric}"
TOKEN="${DOCKER_INFLUXDB_INIT_ADMIN_TOKEN:-ric_admin_token}"

# Help message
function show_help {
    echo "Usage: $0 [OPTIONS]"
    echo "Backup the InfluxDB network_metrics bucket"
    echo
    echo "Options:"
    echo "  -d, --backup-dir DIR   Backup directory (default: /backups/YYYYMMDD_HHMMSS)"
    echo "  -b, --bucket BUCKET    Bucket name (default: network_metrics)"
    echo "  -o, --org ORG          Organization name (default: ric)"
    echo "  -t, --token TOKEN      Admin token (default: from environment)"
    echo "  -h, --help             Display this help message and exit"
    echo
    echo "Example:"
    echo "  $0 --backup-dir /backups/my_backup"
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

# Create backup directory if it doesn't exist
echo "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Run the backup command
echo "Starting backup of bucket '$BUCKET' to '$BACKUP_DIR'..."
influx backup "$BACKUP_DIR" \
    --bucket "$BUCKET" \
    --org "$ORG" \
    --token "$TOKEN"

# Check if backup was successful
if [ $? -eq 0 ]; then
    echo "Backup completed successfully to: $BACKUP_DIR"
    # Create a metadata file with backup information
    echo "Creating backup metadata file..."
    cat > "$BACKUP_DIR/backup_metadata.txt" << EOF
Backup Date: $(date)
Bucket: $BUCKET
Organization: $ORG
InfluxDB Version: $(influx version | head -n 1)
EOF
    echo "Backup process complete."
    echo "To restore this backup, use: influxdb_restore.sh --backup-dir $BACKUP_DIR"
else
    echo "Backup failed!"
    exit 1
fi
