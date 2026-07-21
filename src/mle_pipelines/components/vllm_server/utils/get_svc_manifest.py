def get_svc_manifest(
    service_name: str,
    namespace: str,
    labels: dict,
) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": service_name,
            "namespace": namespace,
        },
        "spec": {
            "type": "ClusterIP",
            "selector": labels,
            "ports": [
                {
                    "name": "http",
                    "protocol": "TCP",
                    "port": 8000,
                    "targetPort": "http",
                }
            ],
        },
    }