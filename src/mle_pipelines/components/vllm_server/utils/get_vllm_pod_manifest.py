def get_vllm_pod_manifest(
    pod_name: str,
    labels: dict,
    namespace: str,
    service_account_name: str,
    hf_secret_name: str,
    node_selector_key: str,
    vllm_image: str,
    script: str,
    gpus: int,
    model: str,
    validated_model_config,
    is_local_model: bool,
    tier1_storage_class: str,
    tier1_size_gi: int,
    sidecar_spec: dict,
) -> dict:
    volume_mounts = [
        {"name": "logs", "mountPath": "/logs"},
        {"name": "cache", "mountPath": "/cache"},
        {"name": "cache", "mountPath": "/home", "subPath": "home"},
    ]

    volumes = [
        {"name": "logs", "emptyDir": {}},
        {"name": "cache", "emptyDir": {}},
    ]

    if not is_local_model:
        volumes.append({
            "name": "tier1",
            "ephemeral": {
                "volumeClaimTemplate": {
                    "metadata": {
                        "labels": {"type": "ephemeral-volume"},
                    },
                    "spec": {
                        "accessModes": ["ReadWriteOnce"],
                        "storageClassName": tier1_storage_class,
                        "resources": {
                            "requests": {
                                "storage": f"{tier1_size_gi}Gi",
                            },
                        },
                    },
                },
            },
        })
        volume_mounts.append({"name": "tier1", "mountPath": "/tier1"})

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
            "securityContext": {
                "fsGroup": 1000770000,
                "fsGroupChangePolicy": "Always",
            },
            "nodeSelector": {
                node_selector_key: "",
            },
            "containers": [
                {
                    "name": "vllm-server",
                    "image": vllm_image,
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
                            "path": "/v1/models",
                            "port": "http",
                        },
                        "initialDelaySeconds": 30,
                        "periodSeconds": 15,
                        "timeoutSeconds": 10,
                        "failureThreshold": 120,
                    },
                    "resources": {
                        "requests": {
                            "nvidia.com/gpu": str(gpus),
                        },
                        "limits": {
                            "nvidia.com/gpu": str(gpus),
                        },
                    },
                    "env": [
                        {"name": "MODEL", "value": model},
                        {"name": "IS_LOCAL_MODEL", "value": "true" if is_local_model else "false"},
                        {"name": "LOGS_PATH", "value": "/logs/vllm_serve.log"},
                        {"name": "TP", "value": str(validated_model_config.tp)},
                        {"name": "DP", "value": str(validated_model_config.dp)},
                        {"name": "MAX_MODEL_LEN", "value": str(validated_model_config.max_model_length)},
                        {"name": "REASONING_PARSER", "value": validated_model_config.reasoning_parser},
                        {
                            "name": "HF_TOKEN",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": hf_secret_name,
                                    "key": "HF_TOKEN",
                                }
                            },
                        },
                        {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
                        {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"},
                        {"name": "RAYON_NUM_THREADS", "value": "1"},
                        {"name": "VLLM_ENGINE_READY_TIMEOUT_S", "value": "1800"},

                        {"name": "HOME", "value": "/home"},
                        {"name": "USER", "value": "vllm"},
                        {"name": "LOGNAME", "value": "vllm"},
                        {"name": "XDG_CACHE_HOME", "value": "/cache"},
                        {"name": "TORCH_HOME", "value": "/cache/torch"},
                        {"name": "TORCHINDUCTOR_CACHE_DIR", "value": "/cache/torchinductor"},
                        {"name": "TRITON_CACHE_DIR", "value": "/cache/triton"},
                        {"name": "VLLM_CACHE_ROOT", "value": "/cache/vllm"},
                        {"name": "VLLM_USE_V2_MODEL_RUNNER", "value": "0"},
                    ],
                    "envFrom": [
                        {"configMapRef": {"name": "ceph-bucket-class"}},
                        {"secretRef": {"name": "ceph-bucket-class"}},
                    ],
                    "volumeMounts": volume_mounts,
                },
                sidecar_spec,
            ],
            "volumes": volumes,
        },
    }

    return manifest
