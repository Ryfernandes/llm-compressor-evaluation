def get_proxy_pod_manifest(
    pod_name: str,
    labels: dict,
    namespace: str,
    service_account_name: str,
    script: str,
    vllm_service_url: str,
    run_id: str,
    sidecar_spec: dict,
) -> dict:
    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": namespace,
            "labels": labels,
        },
        "spec": {
            "restartPolicy": "Never",
            "serviceAccountName": service_account_name,
            "terminationGracePeriodSeconds": 120,
            "containers": [
                {
                    "name": "vllm-proxy",
                    "image": "python:3.12",
                    "ports": [
                        {
                            "name": "http",
                            "containerPort": 8000,
                            "protocol": "TCP",
                        }
                    ],
                    "command": ["/bin/bash", "-c", script],
                    "readinessProbe": {
                        "httpGet": {
                            "path": "/health",
                            "port": "http",
                        },
                        "initialDelaySeconds": 10,
                        "periodSeconds": 5,
                        "timeoutSeconds": 3,
                        "failureThreshold": 20,
                    },
                    "env": [
                        {"name": "VLLM_SERVICE_URL", "value": vllm_service_url},
                        {"name": "LOGS_PATH", "value": "/logs/vllm_requests.jsonl"},
                    ],
                    "envFrom": [
                        {"configMapRef": {"name": "ceph-bucket-class"}},
                        {"secretRef": {"name": "ceph-bucket-class"}},
                    ],
                    "volumeMounts": [
                        {"name": "logs", "mountPath": "/logs"},
                    ],
                },
                sidecar_spec,
            ],
            "volumes": [
                {
                    "name": "logs",
                    "emptyDir": {},
                },
            ],
        },
    }

    return manifest
