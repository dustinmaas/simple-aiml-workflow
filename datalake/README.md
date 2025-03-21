# DataLake Management

Tools for backing up and restoring InfluxDB metrics data.

## Backup and Restore Commands

### Backup

```bash
./datalake/backup_influxdb_metrics.sh [options]
```

### Restore

```bash
./datalake/restore_influxdb_metrics.sh [options]
```

## Common Options

- `-b, --bucket BUCKET`: Bucket name (default: network_metrics)
- `-o, --org ORG`: Organization name (default: ric)
- `-h, --help`: Display help message and exit

## Backup-Specific Options

- `-d, --backup-dir DIR`: Directory to store the backup (default: ./datalake/backups/YYYYMMDD_HHMMSS)

## Restore-Specific Options

- `-d, --backup-dir DIR`: Backup directory to restore from (default: ./datalake/backups/latest)

## Examples

```bash
# Backup to default location
./datalake/backup_influxdb_metrics.sh

# Backup to custom location
./datalake/backup_influxdb_metrics.sh --backup-dir ./datalake/backups/custom_backup

# Restore from latest backup
./datalake/restore_influxdb_metrics.sh

# Restore from specific backup
./datalake/restore_influxdb_metrics.sh --backup-dir ./datalake/backups/20250318_123456
```

## What These Scripts Do

Both scripts will:

1. Check if the InfluxDB container is running (starting it if needed)
1. Execute the backup/restore operation
1. Create a 'latest' symlink to the most recent backup (backup only)
1. Stop the container if it wasn't running before the operation
