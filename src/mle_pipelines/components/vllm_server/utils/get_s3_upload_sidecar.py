def get_s3_upload_sidecar_spec(
    log_file_path: str,
    s3_key: str,
) -> dict:
    script = r'''
set -eu

echo "[log-uploader] Installing boto3..."
pip install --no-cache-dir boto3 > /dev/null 2>&1
echo "[log-uploader] Ready. Waiting for shutdown signal..."

upload_logs() {
    echo "[log-uploader] Shutdown signal received. Waiting for logs to flush..."
    sleep 10

    if [ ! -s "${LOG_FILE_PATH}" ]; then
        echo "[log-uploader] WARNING: Log file ${LOG_FILE_PATH} is empty or missing. Skipping upload."
        exit 0
    fi

    python3 -c "
import boto3
import os

s3 = boto3.client(
    's3',
    endpoint_url=os.environ['BUCKET_HOST'],
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
)

log_file = os.environ['LOG_FILE_PATH']
bucket = os.environ['BUCKET_NAME']
key = os.environ['S3_KEY']

s3.upload_file(log_file, bucket, key)
print(f'[log-uploader] Uploaded {log_file} -> s3://{bucket}/{key}')
"
    echo "[log-uploader] Upload complete."
    exit 0
}

trap upload_logs SIGTERM

while true; do
    sleep 1 &
    wait $!
done
'''

    return {
        "name": "log-uploader",
        "image": "python:3.12-slim",
        "command": ["/bin/bash", "-c", script],
        "env": [
            {"name": "LOG_FILE_PATH", "value": log_file_path},
            {"name": "S3_KEY", "value": s3_key},
        ],
        "envFrom": [
            {"configMapRef": {"name": "pipelines-ceph"}},
            {"secretRef": {"name": "pipelines-ceph"}},
        ],
        "volumeMounts": [
            {"name": "logs", "mountPath": "/logs"},
        ],
        "resources": {
            "requests": {
                "cpu": "100m",
                "memory": "128Mi",
            },
            "limits": {
                "cpu": "500m",
                "memory": "256Mi",
            },
        },
    }
