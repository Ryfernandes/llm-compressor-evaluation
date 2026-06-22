import kfp
from kfp import dsl, kubernetes
from collate_results_component import collate_results
from github_upload_component import upload_results_to_github

PIPELINE_NAME = "all-evals-pipeline"

"""
Pipeline Spec

- Create session ID

In parallel:

- Pull model and start vLLM server
- Run test inference
- Run full evals
- Tear down server

- Run compression with LLM compressor
- Start vLLM server with compressed model
- Run test inference
- Run full evals
- Tear down server

- Collate results and save to tier 2
- Upload from tier 2 to GitHub
"""

@dsl.component()
def generate_session_id() -> str:
    from datetime import datetime, timezone
    import uuid

    timestamp = datetime.now(timezone.utc).strftime("%y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{timestamp}-{suffix}"

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

exec vllm serve "$LOCAL_MODEL" \
  --served-model-name "$MODEL" \
  --tensor-parallel-size "${TP}" \
  --data-parallel-size "${DP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --host 0.0.0.0 \
  --port 8000
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
                        {"name": "TIER2_MODEL_STAGING_PATH", "value": tier2_model_staging_path},
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

@kfp.dsl.component(
    base_image="quay.io/opendatahub/llmcompressor-pipeline-runtime:main",
)
def compress_model(
    model_id: str, recipe: str, output_model: dsl.Output[dsl.Artifact]
):
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = oneshot(model=model, recipe=recipe, tokenizer=tokenizer)

    model.save_pretrained(output_model.path)
    tokenizer.save_pretrained(output_model.path)

    return

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["requests"],
)
def test_vllm_server(
    service_url: str,
    model: str = "Qwen/Qwen3-8B",
) -> str:
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
                "content": "Briefly explain methods to evaluate large language models.",
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

    print("Test inference successful")

    return service_url

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["lm-eval[api]"]
)
def evaluate_model(
    service_url: str,
    tasks: str,
    session_id: str,
    model_path: str = "Qwen/Qwen3-8B",
    max_length: int = 32768,
    n_shots: int = 5,
    max_gen_toks: int = 4096,
    reps: int = 1,
    save_path: str = "/tier2/evaluations",
    save_prefix: str = "baseline",
) -> None:
    import os
    import subprocess
    from pathlib import Path

    # Parse and clean task list
    task_list = [task.strip() for task in tasks.split(",") if task.strip()]

    # Setup paths
    session_dir = Path(save_path) / "sessions" / session_id / save_prefix
    session_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = session_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    results_dir = session_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Change to session directory
    os.chdir(session_dir)

    # Ensure service_url uses the correct format for base_url
    base_url = service_url.rstrip("/") + "/v1"

    try:
        for task_tag in task_list:
            print(f"\n{'='*60}")
            print(f"Evaluating task: {task_tag}")
            print(f"{'='*60}\n")

            for i in range(1, reps + 1):
                print(f"Evaluation Run {i}/{reps}")

                seed = 1233 + i

                # Build lm_eval command
                cmd = [
                    "python", "-m", "lm_eval",
                    "--model", "local-chat-completions",
                    "--tasks", task_tag,
                    "--model_args", (
                        f"model={model_path},"
                        f"max_length={max_length},"
                        f"base_url={base_url}/chat/completions,"
                        f"num_concurrent=128,"
                        f"max_retries=3,"
                        f"tokenized_requests=False,"
                        f"tokenizer_backend=None,"
                        f"timeout=1200"
                    ),
                    "--num_fewshot", str(n_shots),
                    "--apply_chat_template",
                    "--fewshot_as_multiturn",
                    "--output_path", str(tmp_dir),
                    "--seed", str(seed),
                    "--gen_kwargs", (
                        f"do_sample=True,"
                        f"temperature=0.6,"
                        f"top_p=0.9,"
                        f"top_k=50,"
                        f"max_gen_toks={max_gen_toks},"
                        f"seed={seed}"
                    ),
                ]

                # Run evaluation
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)

                print(f"Evaluation complete, tried moving output to {str(tmp_dir)}")

                json_files = list(tmp_dir.rglob("*.json"))

                if json_files:
                    # Use the first (or only) results file found
                    json_file = json_files[0]
                    output_name = f"{task_tag}_seed_{seed}.json"
                    output_path = results_dir / output_name

                    # Copy instead of rename to avoid cross-device link errors
                    import shutil
                    shutil.copy2(json_file, output_path)
                    print(f"Saved results to: {output_path}")
                    print(f"Source file: {json_file}")
                else:
                    print(f"WARNING: No JSON output found for {task_tag} run {i}")
                    # Debug: list what's actually in tmp_dir
                    print(f"Contents of {tmp_dir}:")
                    for item in tmp_dir.rglob("*"):
                        print(f"  {item}")

                # Clean tmp directory
                for file in tmp_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        import shutil
                        shutil.rmtree(file)

        print(f"\n{'='*60}")
        print(f"All evaluations complete. Results saved to: {results_dir}")
        print(f"{'='*60}")

    finally:
        # Cleanup tmp directory
        if tmp_dir.exists():
            import shutil
            shutil.rmtree(tmp_dir)

@dsl.component(
    base_image="python:3.12",
)
def cleanup_model_staging(
    session_id: str,
    tier2_mount_path: str = "/tier2",
) -> None:
    import shutil
    from pathlib import Path

    tier2_staging_dir = Path(tier2_mount_path) / "model-staging" / session_id

    if tier2_staging_dir.exists():
        print(f"[*] Cleaning up tier2 staging directory: {tier2_staging_dir}")
        shutil.rmtree(tier2_staging_dir)
        print(f"[+] Tier2 staging directory removed")
    else:
        print(f"[*] Tier2 staging directory does not exist (already cleaned up): {tier2_staging_dir}")

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
    name="full-eval-pipeline"
)
def pipeline(
    model_id: str = "Qwen/Qwen3-8B",
    compression_recipe: str = """
        quant_stage:
            quant_modifiers:
                QuantizationModifier:
                    ignore: ["lm_head"]
                    targets: ["Linear"]
                    scheme: "W4A16"
        """,
    evaluation_tasks: str = "gsm8k_platinum_cot_llama",
):
    session_id_task = generate_session_id()
    session_id_task.set_caching_options(enable_caching=False)

    datafree_recipe = """
        quant_stage:
            quant_modifiers:
                QuantizationModifier:
                    ignore: ["lm_head"]
                    targets: ["Linear"]
                    scheme: "W4A16"
    """

    """
    cleanup_task = delete_vllm_server(
        session_id=session_id,
        delete_tier2_pvc=False,
    )
    cleanup_task.set_caching_options(enable_caching=False)
    
    with dsl.ExitHandler(cleanup_task):
    
    """
    
    ### Baseline model flow
    create_vllm_task = create_vllm_server(session_id=session_id_task.output, pod_suffix="b", model=model_id)
    create_vllm_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        create_vllm_task,
        pvc_name="evaluation-pipeline-model-server-tier-2",
        mount_path="/tier2"
    )
    
    test_vllm_task = test_vllm_server(service_url=create_vllm_task.output, model=model_id)
    test_vllm_task.set_caching_options(enable_caching=False)

    evaluation_task = (
        evaluate_model(session_id=session_id_task.output, service_url=test_vllm_task.output, tasks=evaluation_tasks, save_prefix="baseline", model_path=model_id)
        .set_accelerator_type("nvidia.com/gpu")
        .set_accelerator_limit("1")
    )
    evaluation_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        evaluation_task,
        pvc_name="evaluation-pipeline-artifacts-tier-2",
        mount_path="/tier2"
    )
    delete_vllm_task = (
        delete_vllm_server(
            session_id=session_id_task.output,
            delete_tier2_pvc=False,
            pod_suffix="b",
        )
        .after(evaluation_task)
    )
    delete_vllm_task.set_caching_options(enable_caching=False)

    ### Compressed model flow
    quantization_task = compress_model(model_id=model_id, recipe=compression_recipe)

    quantized_create_vllm_task = create_vllm_server(session_id=session_id_task.output, input_model=quantization_task.outputs["output_model"], pod_suffix="c")
    quantized_create_vllm_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        quantized_create_vllm_task,
        pvc_name="evaluation-pipeline-model-server-tier-2",
        mount_path="/tier2"
    )
    
    quantized_test_vllm_task = test_vllm_server(service_url=quantized_create_vllm_task.output, model=model_id)
    quantized_test_vllm_task.set_caching_options(enable_caching=False)
    
    quantized_cleanup_staging_task = cleanup_model_staging(session_id=session_id_task.output).after(quantized_test_vllm_task)
    quantized_cleanup_staging_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        quantized_cleanup_staging_task,
        pvc_name="evaluation-pipeline-model-server-tier-2",
        mount_path="/tier2"
    )

    quantized_evaluation_task = (
        evaluate_model(session_id=session_id_task.output, service_url=quantized_test_vllm_task.output, tasks=evaluation_tasks, save_prefix="compressed", model_path=model_id)
        .set_accelerator_type("nvidia.com/gpu")
        .set_accelerator_limit("1")
    )
    quantized_evaluation_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        quantized_evaluation_task,
        pvc_name="evaluation-pipeline-artifacts-tier-2",
        mount_path="/tier2"
    )
    quantized_delete_vllm_task = (
        delete_vllm_server(
            session_id=session_id_task.output,
            delete_tier2_pvc=False,
            pod_suffix="c",
        )
        .after(quantized_evaluation_task)
    )
    quantized_delete_vllm_task.set_caching_options(enable_caching=False)

    ### Collate results from both flows
    collate_task = (
        collate_results(session_id=session_id_task.output)
        .after(delete_vllm_task, quantized_delete_vllm_task)
    )
    collate_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        collate_task,
        pvc_name="evaluation-pipeline-artifacts-tier-2",
        mount_path="/tier2"
    )

    ### Upload collated results to GitHub
    github_upload_task = upload_results_to_github(session_id=session_id_task.output).after(collate_task)
    github_upload_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        github_upload_task,
        pvc_name="evaluation-pipeline-artifacts-tier-2",
        mount_path="/tier2"
    )
    kubernetes.use_secret_as_env(
        github_upload_task,
        secret_name="evaluation-pipeline-results-gh",
        secret_key_to_env={"GITHUB_TOKEN": "GITHUB_TOKEN"}
    )

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, package_path=f"{PIPELINE_NAME}.yaml"
    )