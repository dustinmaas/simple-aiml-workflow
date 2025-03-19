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
3. Run the backup inside the container
4. Copy the backup data from the container to the host
5. Stop the container if it wasn't running before

## Restore Usage

To restore the InfluxDB network_metrics bucket:

```bash
./datalake/restore_influxdb_metrics.sh [options]
```

### Options:

- `-d, --backup-dir DIR`: Backup directory (default: ./datalake/backups/latest)
- `-b, --bucket BUCKET`: Bucket name (default: network_metrics)
- `-o, --org ORG`: Organization name (default: ric)
- `-f, --force`: Force restore by deleting existing bucket first
- `-h, --help`: Display help message and exit

### Example:

```bash
# Basic restore using default options (latest backup)
./datalake/restore_influxdb_metrics.sh

# Restore from specific backup
./datalake/restore_influxdb_metrics.sh --backup-dir ./datalake/backups/20250318_123456

# The script will check if the bucket exists and ask for confirmation
./datalake/restore_influxdb_metrics.sh
```

This script will:
1. Check if the InfluxDB container is running and start it if needed
2. Use the backup directly from the mounted directory if possible
3. Run the restore script already mounted in the container
4. Stop the container if it wasn't running before

### Interactive Bucket Check

When restoring, the script automatically checks if the target bucket already exists:
1. If the bucket doesn't exist, the restore proceeds automatically
2. If the bucket already exists, you'll be prompted for confirmation:
   ```
   WARNING: Bucket 'network_metrics' already exists (ID: 4b8391eba2c60c26)
   Restoring will overwrite all existing data in this bucket.

   Delete existing bucket and continue with restore? (y/N):
   ```
3. Enter 'y' to delete the existing bucket and proceed, or any other key to cancel

### Latest Backup Symlink

The restore script automatically creates and uses a `latest` symlink in the backups directory, which points to the most recent backup. This allows you to easily restore the most recent backup without specifying a directory:

```bash
# Create the latest symlink (if needed)
ln -sf $(ls -td datalake/backups/202* | head -1) datalake/backups/latest

# Restore from the latest backup
./datalake/restore_influxdb_metrics.sh
```
