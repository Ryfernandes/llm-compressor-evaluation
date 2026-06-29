from kfp import dsl

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["kubernetes", "requests"]
)
def create_vllm_server(
    session_id: str,
    reasoning_parser: str,
    namespace: str = "machine-learning",
    pod_suffix: str = "baseline-server",
    service_account_name: str = "ml-workload",
    model: str = "Qwen/Qwen3-8B",
    input_model: dsl.Input[dsl.Artifact] = None,
    hf_secret_name: str = "ryan-test-hf-hub-secret",
    tier2_pvc_name: str = "evaluation-pipeline-model-server-tier-2",
    tier2_mount_path: str = "/tier2",
    fs_group: int = 1000770000,
    max_model_len: int = 32768,
    tp: str = "1",
    dp: str = "1",
    tier1_storage_class: str = "lvms-h100-tier1-storage",
    tier1_size: str = "50Gi",
    node_selector_key: str = "node-role.kubernetes.io/up-h100mcp",
    wait_timeout_seconds: int = 2700,
) -> str:
    import time
    import requests
    import os
    from pathlib import Path
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException

    config.load_incluster_config()
    core = client.CoreV1Api()

    pod_name = f"{namespace}--{session_id}--{pod_suffix}"
    service_name = f"{pod_name}-svc"

    labels = {
        "app.kubernetes.io/instance": pod_name,
    }

    service_url = f"http://{service_name}.{namespace}.svc.cluster.local:8000"

    # Determine if we're using a local model artifact or HuggingFace
    use_local_artifact = input_model is not None
    local_artifact_path = input_model.path if use_local_artifact else ""

    # Copy artifact to tier2 staging area for the pod to access
    tier2_model_staging_path = ""
    if use_local_artifact:
        print(f"[*] Artifact path: {local_artifact_path}")
        artifact_path = Path(local_artifact_path)

        if artifact_path.exists():
            print(f"[+] Artifact path exists and is accessible from this component")

            if artifact_path.is_dir():
                files = list(artifact_path.glob("*"))
                print(f"[+] Artifact directory contains {len(files)} items")

                # Copy to tier2 staging area
                tier2_staging_dir = Path(tier2_mount_path) / "model-staging" / session_id / pod_suffix
                tier2_staging_dir.mkdir(parents=True, exist_ok=True)

                print(f"[*] Copying model artifact to tier2 staging: {tier2_staging_dir}")
                import shutil

                # Copy all files from artifact to staging
                for item in artifact_path.iterdir():
                    dest = tier2_staging_dir / item.name
                    if item.is_file():
                        shutil.copy2(item, dest)
                    else:
                        shutil.copytree(item, dest, dirs_exist_ok=True)

                tier2_model_staging_path = str(tier2_staging_dir)
                print(f"[+] Model copied to tier2 staging: {tier2_model_staging_path}")

                # Verify
                copied_files = list(tier2_staging_dir.glob("*"))
                print(f"[+] Verified {len(copied_files)} items in staging area")
            else:
                print(f"[!] Artifact path is a file, not a directory")
                raise ValueError(f"Expected artifact directory, got file: {local_artifact_path}")
        else:
            print(f"[!] ERROR: Artifact path does not exist: {local_artifact_path}")
            raise FileNotFoundError(f"Artifact path not accessible: {local_artifact_path}")

    if use_local_artifact:
        script = r'''
set -eu

export HF_HOME="/tier2/hf-hub"
export HF_HUB_CACHE="/tier2/hf-hub"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p \
  "$HF_HOME" \
  "$HOME" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$VLLM_CACHE_ROOT"

LOCAL_MODEL="/tier1/model"
TIER2_STAGING="${TIER2_MODEL_STAGING_PATH}"

echo "[*] Copying model from tier2 staging to tier1 NVMe..."
echo "[*] Source: ${TIER2_STAGING}"
echo "[*] Destination: ${LOCAL_MODEL}"

if [ ! -d "${TIER2_STAGING}" ]; then
    echo "[!] ERROR: Tier2 staging directory does not exist: ${TIER2_STAGING}"
    exit 1
fi

rm -rf "$LOCAL_MODEL"
mkdir -p "$LOCAL_MODEL"

# Fast local copy from tier2 to tier1
echo "[*] Starting copy..."
cp -rv "${TIER2_STAGING}"/. "$LOCAL_MODEL"/

echo "[+] Model copied to ${LOCAL_MODEL}"
echo "[*] Verifying files:"
ls -lh "$LOCAL_MODEL"
echo "[*] Total files:"
find "$LOCAL_MODEL" -type f | wc -l

echo "================================================================"
echo " Starting vLLM Server"
echo " Model name: ${MODEL}"
echo " Source: tier2 staging (${TIER2_STAGING})"
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

exec vllm serve "${VLLM_ARGS[@]}"
'''
    else:
        script = r'''
set -eu

export HF_HOME="/tier2/hf-hub"
export HF_HUB_CACHE="/tier2/hf-hub"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p \
  "$HF_HOME" \
  "$HOME" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$VLLM_CACHE_ROOT"

LOCAL_MODEL="/tier1/model"

echo "[*] Resolving/downloading model from Hugging Face..."
CACHE_PATH=$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('${MODEL}'))
")
echo "[+] HF cache path: ${CACHE_PATH}"

echo "[*] Copying model to tier1 NVMe..."
rm -rf "$LOCAL_MODEL"
mkdir -p "$LOCAL_MODEL"
cp -rL "$CACHE_PATH"/. "$LOCAL_MODEL"/
echo "[+] Model copied to ${LOCAL_MODEL}"

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

exec vllm serve "${VLLM_ARGS[@]}"
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
                "fsGroup": fs_group,
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
                            "nvidia.com/gpu": "1",
                        },
                        "limits": {
                            "nvidia.com/gpu": "1",
                        },
                    },
                    "env": [
                        {"name": "MODEL", "value": model},
                        {"name": "TP", "value": tp},
                        {"name": "DP", "value": dp},
                        {"name": "MAX_MODEL_LEN", "value": str(max_model_len)},
                        {"name": "TIER2_MODEL_STAGING_PATH", "value": tier2_model_staging_path},
                        {"name": "REASONING_PARSER", "value": reasoning_parser},
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
                        "claimName": tier2_pvc_name,
                    },
                },
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
    
    create_or_patch_service()

    delete_pod_if_exists()
    create_pod()

    wait_for_pod_ready()
    wait_for_service_healthy()

    print(f"vLLM service URL: {service_url}")
    return service_url