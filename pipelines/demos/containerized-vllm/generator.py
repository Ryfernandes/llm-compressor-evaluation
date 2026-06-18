import kfp
from kfp import dsl, kubernetes

PIPELINE_NAME = "containerized-vllm-pipeline"

@dsl.component(
    base_image="python:3.12",
)
def clean_session_id(session_id: str) -> str:
    return session_id.replace("_", "-").lower()

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["kubernetes", "requests"]
)
def create_vllm_server(
    session_id: str,
    namespace: str = "machine-learning",
    pod_suffix: str = "baseline-server",
    service_account_name: str = "ml-workload",
    model: str = "Qwen/Qwen3-8B",
    hf_secret_name: str = "ryan-test-hf-hub-secret",
    tier2_pvc_name: str = "evaluation-pipeline-model-server-tier-2",
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

exec vllm serve "$LOCAL_MODEL" \
  --served-model-name "$MODEL" \
  --tensor-parallel-size "${TP}" \
  --data-parallel-size "${DP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --host 0.0.0.0 \
  --port 8000
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

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["requests"],
)
def test_vllm_server(
    service_url: str,
    prompt: str,
    model: str = "Qwen/Qwen3-8B",
) -> None:
    import json
    import requests

    models_url = f"{service_url}/v1/models"
    print(f"Checking {models_url}")

    models_response = requests.get(models_url, timeout=30)
    models_response.raise_for_status()
    print(json.dumps(models_response.json(), indent=2))

    chat_url = f"{service_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.6,
        "chat_template_kwargs": {
            "enable_thinking": True,
        },
    }

    response = requests.post(chat_url, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    print(json.dumps(data, indent=2))
    print("Assistant response:")
    print(data["choices"][0]["message"]["content"])

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["kubernetes"]
)
def delete_vllm_server(
    session_id: str,
    namespace: str = "machine-learning",
    pod_suffix: str = "baseline-server",
    tier2_pvc_name: str = "evaluation-pipeline-model-server-tier-2",
    delete_tier2_pvc: bool = False,
) -> None:
    import time
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException

    config.load_incluster_config()
    core = client.CoreV1Api()

    pod_name = f"{namespace}--{session_id}--{pod_suffix}"
    service_name = f"{pod_name}-svc"

    def ignore_404(fn, description: str) -> None:
        try:
            fn()
            print(f"Deleted {description}")
        except ApiException as e:
            if e.status == 404:
                print(f"{description} did not exist")
            else:
                raise
    
    ignore_404(
        lambda: core.delete_namespaced_pod(name=pod_name, namespace=namespace),
        f"pod/{pod_name}",
    )

    ignore_404(
        lambda: core.delete_namespaced_service(name=service_name, namespace=namespace),
        f"service/{service_name}",
    )

    tier1_pvc_name = f"{pod_name}-tier1"
    ignore_404(
        lambda: core.delete_namespaced_persistent_volume_claim(
            name=tier1_pvc_name,
            namespace=namespace,
        ),
        f"pvc/{tier1_pvc_name}",
    )

    if delete_tier2_pvc:
        ignore_404(
            lambda: core.delete_namespaced_persistent_volume_claim(
                name=tier2_pvc_name,
                namespace=namespace,
            ),
            f"pvc/{tier2_pvc_name}",
        )
    else:
        print(f"Preserved tier2 PV: {tier2_pvc_name}")

    # Wait for pod deletion to complete
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            core.read_namespaced_pod(name=pod_name, namespace=namespace)
            print("Waiting for pod deletion...")
            time.sleep(5)
        except ApiException as e:
            if e.status == 404:
                print("Pod deletion complete")
                return
            raise

    print("Timed out waiting for pod deletion, but delete was requested")

@dsl.pipeline(
    name="managed-permanent-vllm-server"
)
def pipeline(
    session_id: str,
    prompt: str = "Concisely and formally prove that there are infinitely many prime numbers.",
):
    cleanup_task = delete_vllm_server(
        session_id=session_id,
        delete_tier2_pvc=False,
    )
    cleanup_task.set_caching_options(enable_caching=False)
    
    clean_session_id_task = clean_session_id(session_id=session_id)

    with dsl.ExitHandler(cleanup_task):
        create_vllm_task = create_vllm_server(session_id=clean_session_id_task.output)
        create_vllm_task.set_caching_options(enable_caching=False)
        test_vllm_task = test_vllm_server(service_url=create_vllm_task.output, prompt=prompt)
        test_vllm_task.set_caching_options(enable_caching=False)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path=f"{PIPELINE_NAME}.yaml"
    )