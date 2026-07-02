from kfp import dsl
from typing import Optional

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["kubernetes", "requests", "huggingface-hub"]
)
def create_vllm_server(
    # Model spec
    model: str,
    # Session spec
    session_id: str,
    # Evaluation spec
    config_filename: str,
    # PVC spec
    model_server_pvc_name: str,
    configs_pvc_mount_path: str,
    artifacts_pvc_name: str,
    artifacts_pvc_mount_path: str,
    # Pod spec
    namespace: str = "machine-learning",
    service_account_name: str = "ml-workload",
    hf_secret_name: str = "ryan-test-hf-hub-secret",
    tier1_storage_class: str = "lvms-h100-tier1-storage",
    tier1_storage_buffer_gi: int = 10,
    kv_cache_buffer_gi: int = 20,
    gib_per_gpu: int = 76,
    node_selector_key: str = "node-role.kubernetes.io/up-h100mcp",
    # Timeout spec
    wait_timeout_seconds: int = 2700,
    # Logs spec
    logs_filename: str = "vllm_server.log",
    evaluation_statistics_filename: str = "evaluation_statistics.json",
) -> str:
    import time
    import requests
    import os
    import math
    from pathlib import Path
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    from huggingface_hub import snapshot_download
    import json

    config.load_incluster_config()
    core = client.CoreV1Api()

    # Load configuration (already validated by validate_config component)
    evaluation_config_path = Path(configs_pvc_mount_path) / config_filename

    with open(evaluation_config_path, "r") as f:
        evaluation_config = json.load(f)
        model_args = evaluation_config["model"]

        reasoning_parser = model_args.get("reasoning_parser", "")
        max_model_len = model_args["max_model_len"]
        tp = model_args.get("tp", 1)
        dp = model_args.get("dp", 1)

    # Dry run download to estimate model size
    print(f"Running dry run to estimate size of model {model}...")

    dry_run_files = snapshot_download(
        repo_id=model,
        token=os.environ.get("HF_TOKEN"),
        dry_run=True,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "*.jinja",
            "tokenizer*",
            "special_tokens_map.json",
        ],
        ignore_patterns=[
            "original/**/*",
            "*.bin",
            "*.gguf",
        ],
    )

    model_size_bytes = sum(
        f.file_size for f in dry_run_files
        if f.file_size is not None
    )

    model_size_gi = model_size_bytes / (1024 ** 3)
    tier1_size_gi = int(model_size_gi) + tier1_storage_buffer_gi + 1
    tier1_size = f"{tier1_size_gi}Gi"
    inference_size_gi = model_size_gi + kv_cache_buffer_gi

    print(f"Model size: {model_size_gi:.2f} GiB")
    print(f"Tier1 storage size (with {tier1_storage_buffer_gi} GiB buffer): {tier1_size}")
    print(f"Inference size (with {kv_cache_buffer_gi} GiB minimum KV cache buffer): {inference_size_gi:.2f} GiB")

    # Validate and adjust tp based on model size
    min_tp = math.ceil(inference_size_gi / gib_per_gpu)
    if tp < min_tp:
        # Round up to next power of 2
        adjusted_tp = 2 ** math.ceil(math.log2(min_tp))
        print(f"WARNING: tp={tp} is too low for model size {model_size_gi:.2f} GiB")
        print(f"WARNING: Minimum tp based on {gib_per_gpu} GiB/GPU is {min_tp}")
        print(f"WARNING: Increasing tp from {tp} to {adjusted_tp} (next power of 2)")
        tp = adjusted_tp

    gpus = tp * dp
    print(f"Final configuration: tp={tp}, dp={dp}, gpus={gpus}")

    session_path = Path(artifacts_pvc_mount_path) / "evaluation-artifacts" / "sessions" / session_id
    if not session_path.exists():
        raise FileNotFoundError(f"Session path {session_path} does not exist. Ensure validate_session_id has been run.")
    
    logs_path = session_path / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)

    pod_name = f"evals-vllm-server-{session_id}"
    service_name = f"{pod_name}-svc"

    labels = {
        "app.kubernetes.io/instance": pod_name,
    }

    service_url = f"http://{service_name}.{namespace}.svc.cluster.local:8000"

    script = r'''
set -eu

export HF_HOME="/tier1/hf-hub"
export HF_HUB_CACHE="/tier1/hf-hub"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p \
  "$HOME" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$VLLM_CACHE_ROOT" \
  "$HF_HOME"

LOCAL_MODEL="/tier1/model"

echo "[*] Downloading model ${MODEL} directly to tier1..."
python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='${MODEL}',
    cache_dir='${HF_HOME}',
    token=os.environ.get('HF_TOKEN'),
    local_dir='${LOCAL_MODEL}',
    local_dir_use_symlinks=False,
)
"
echo "[+] Model downloaded to ${LOCAL_MODEL}"

echo "================================================================"
echo " Starting vLLM Server"
echo " HF model: ${MODEL}"
echo " Local path: ${LOCAL_MODEL}"
echo " TP=${TP}, DP=${DP}"
echo "================================================================"

VLLM_ARGS=(
  "$LOCAL_MODEL"
  --served-model-name "$MODEL"
  --tensor-parallel-size "${TP}"
  --data-parallel-size "${DP}"
  --max-model-len "${MAX_MODEL_LEN}"
  --host 0.0.0.0
  --port 8000
)

if [ -n "${REASONING_PARSER}" ]; then
  VLLM_ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi

exec vllm serve "${VLLM_ARGS[@]}" 2>&1 | tee -a "${LOGS_PATH}"
'''

    pod_manifest = {
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
                    "image": "vllm/vllm-openai:v0.22.1",
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
                        {"name": "TP", "value": str(tp)},
                        {"name": "DP", "value": str(dp)},
                        {"name": "MAX_MODEL_LEN", "value": str(max_model_len)},
                        {"name": "REASONING_PARSER", "value": reasoning_parser},
                        {"name": "LOGS_PATH", "value": str(logs_path / logs_filename)},
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

                        {"name": "HOME", "value": "/tier2/home"},
                        {"name": "USER", "value": "vllm"},
                        {"name": "LOGNAME", "value": "vllm"},
                        {"name": "XDG_CACHE_HOME", "value": "/tier2/cache"},
                        {"name": "TORCH_HOME", "value": "/tier2/cache/torch"},
                        {"name": "TORCHINDUCTOR_CACHE_DIR", "value": "/tier2/cache/torchinductor"},
                        {"name": "TRITON_CACHE_DIR", "value": "/tier2/cache/triton"},
                        {"name": "VLLM_CACHE_ROOT", "value": "/tier2/cache/vllm"},
                    ],
                    "volumeMounts": [
                        {"name": "tier1", "mountPath": "/tier1"},
                        {"name": "tier2", "mountPath": "/tier2"},
                        {"name": "artifacts", "mountPath": artifacts_pvc_mount_path},
                    ],
                    "envFrom": [
                        {"configMapRef": {"name": "ceph-bucket-class"}},
                        {"secretRef": {"name": "ceph-bucket-class"}},
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "tier1",
                    "ephemeral": {
                        "volumeClaimTemplate": {
                            "metadata": {
                                "labels": {
                                    "type": "ephemeral-volume",
                                }
                            },
                            "spec": {
                                "accessModes": ["ReadWriteOnce"],
                                "storageClassName": tier1_storage_class,
                                "resources": {
                                    "requests": {
                                        "storage": tier1_size,
                                    }
                                },
                            },
                        }
                    },
                },
                {
                    "name": "tier2",
                    "persistentVolumeClaim": {
                        "claimName": model_server_pvc_name,
                    },
                },
                {
                    "name": "artifacts",
                    "persistentVolumeClaim": {
                        "claimName": artifacts_pvc_name,
                    },
                }
            ],
        },
    }

    service_manifest = {
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

    def delete_pod_if_exists() -> None:
        try:
            core.delete_namespaced_pod(name=pod_name, namespace=namespace)
            print(f"Deleted existing pod {pod_name}; waiting for deletion...")
        except ApiException as e:
            if e.status != 404:
                raise
            return

        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                core.read_namespaced_pod(name=pod_name, namespace=namespace)
                time.sleep(5)
            except ApiException as e:
                if e.status == 404:
                    return
                raise
        raise TimeoutError(f"Timed out waiting for old pod {pod_name} to delete")

    def create_or_patch_service() -> None:
        try:
            core.create_namespaced_service(namespace=namespace, body=service_manifest)
            print(f"Created service {service_name}")
        except ApiException as e:
            if e.status ==409:
                core.patch_namespaced_service(
                    name=service_name,
                    namespace=namespace,
                    body=service_manifest,
                )
                print(f"Patched existing service {service_name}")
            else:
                raise
    
    def create_pod() -> None:
        core.create_namespaced_pod(namespace=namespace, body=pod_manifest)
        print(f"Created pod {pod_name}")

    def wait_for_pod_ready() -> None:
        deadline = time.time() + wait_timeout_seconds
        
        while time.time() < deadline:
            pod = core.read_namespaced_pod(name=pod_name, namespace=namespace)
            phase = pod.status.phase
            conditions = pod.status.conditions or []

            ready = any(
                condition.type == "Ready" and condition.status == "True"
                for condition in conditions
            )

            print(f"Pod phase={phase}, ready={ready}")

            if ready:
                return

            if phase in {"Failed", "Succeeded"}:
                raise RuntimeError(f"Pod ended before becoming ready: phase={phase}")

            time.sleep(15)
        
        raise TimeoutError(f"Timed out waiting for pod {pod_name} to become ready")
        
    def wait_for_service_healthy() -> None:
        deadline = time.time() + 1800
        url = f"{service_url}/v1/models"

        while time.time() < deadline:
            try:
                response = requests.get(url, timeout=10)
                if response.ok:
                    print(f"Service healthy: {response.text[:500]}")
                    return
                print(f"Service responded with HTTP {response.status_code}: {response.text[:300]}")
            except Exception as e:
                print(f"Service not ready yet: {e}")
            
            time.sleep(10)
        
        raise TimeoutError(f"Timed out waiting for {url}")
    
    def save_startup_statistics() -> None:
        pass
    
    create_or_patch_service()

    delete_pod_if_exists()
    create_pod()

    wait_for_pod_ready()
    wait_for_service_healthy()

    print(f"vLLM service URL: {service_url}")
    return service_url