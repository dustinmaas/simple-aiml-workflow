# Datalake Management

## Backup Usage

To backup the InfluxDB network_metrics bucket:

```bash
./datalake/backup_influxdb_metrics.sh [options]
```

### Options:

- `-d, --backup-dir DIR`: Directory to store the backup (default: ./datalake/backups/YYYYMMDD_HHMMSS)
- `-b, --bucket BUCKET`: Bucket name (default: network_metrics)
- `-o, --org ORG`: Organization name (default: ric)
- `-h, --help`: Display help message and exit

### Example:

```bash
./datalake/backup_influxdb_metrics.sh --backup-dir ./datalake/backups/custom_backup
```

This script will:
1. Check if the InfluxDB container is running and start it if needed
2. Execute the backup script already mounted in the container
3. Stop the container if it wasn't running before

## Restore Usage

To restore the InfluxDB network_metrics bucket:

```bash
./datalake/restore_influxdb_metrics.sh [options]
```

This script will:
1. Check if the InfluxDB container is running and start it if needed
2. Execute the restore script already mounted in the container (prompting for bucket deletion if needed)
3. Stop the container if it wasn't running before

### Options:

- `-d, --backup-dir DIR`: Backup directory (default: ./datalake/backups/latest)
- `-b, --bucket BUCKET`: Bucket name (default: network_metrics)
- `-o, --org ORG`: Organization name (default: ric)
- `-h, --help`: Display help message and exit

### Example:

```bash
# Basic restore using default options (latest backup)
./datalake/restore_influxdb_metrics.sh

# Restore from specific backup
./datalake/restore_influxdb_metrics.sh --backup-dir ./datalake/backups/20250318_123456
```

This script will:
1. Check if the InfluxDB container is running and start it if needed
2. Use the backup directly from the mounted directory if possible
3. Run the restore script already mounted in the container
4. Stop the container if it wasn't running before
