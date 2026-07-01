from kfp import dsl

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["kubernetes", "fastapi", "uvicorn", "httpx"]
)
def create_vllm_proxy(
    # Session spec
    session_id: str,
    # vLLM server URL
    vllm_service_url: str,
    # PVC spec
    artifacts_pvc_name: str,
    artifacts_pvc_mount_path: str,
    # Pod spec
    namespace: str = "machine-learning",
    service_account_name: str = "ml-workload",
    # Timeout spec
    wait_timeout_seconds: int = 600,
    # Logs spec
    proxy_statistics_filename: str = "server_proxy_statistics.jsonl",
) -> str:
    import time
    import os
    from pathlib import Path
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException

    config.load_incluster_config()
    core = client.CoreV1Api()

    session_path = Path(artifacts_pvc_mount_path) / "evaluation-artifacts" / "sessions" / session_id
    if not session_path.exists():
        raise FileNotFoundError(f"Session path {session_path} does not exist. Ensure validate_session_id has been run.")

    logs_path = session_path / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)

    pod_name = f"evals-vllm-proxy-{session_id}"
    service_name = f"{pod_name}-svc"

    labels = {
        "app.kubernetes.io/instance": pod_name,
    }

    service_url = f"http://{service_name}.{namespace}.svc.cluster.local:8000"

    # Python script for the proxy server
    proxy_script = r'''
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Response, HTTPException
import httpx
import uvicorn
import os

app = FastAPI()

VLLM_BASE_URL = os.environ["VLLM_SERVICE_URL"]
LOGS_PATH = Path(os.environ["LOGS_PATH"])

class ProxyLogger:
    def __init__(self, logs_path: Path):
        self.logs_path = logs_path
        self.logs_path.parent.mkdir(parents=True, exist_ok=True)
        self.logging_enabled = False
        self.task_id = None
        self.seed = None

    def set_task(self, task_id: Optional[str], seed: Optional[int]):
        self.task_id = task_id
        self.seed = seed

    def enable_logging(self):
        self.logging_enabled = True

    def disable_logging(self):
        self.logging_enabled = False

    def log_request(
        self,
        request_id: str,
        finish_reason: str,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        total_tokens: Optional[int],
        latency_seconds: float,
    ):
        if not self.logging_enabled:
            return

        log_entry = {
            "request_id": request_id,
            "task_id": self.task_id,
            "seed": self.seed,
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": latency_seconds,
            "timestamp": time.time(),
        }

        with open(self.logs_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

logger = ProxyLogger(LOGS_PATH)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/set-proxy-task")
async def set_proxy_task(request: Request):
    body = await request.json()
    task_id = body.get("task_id")
    seed = body.get("seed")
    logger.set_task(task_id, seed)
    return {
        "status": "success",
        "task_id": logger.task_id,
        "seed": logger.seed
    }

@app.post("/start-logging")
async def start_logging():
    logger.enable_logging()
    return {"status": "success", "logging_enabled": logger.logging_enabled}

@app.post("/stop-logging")
async def stop_logging():
    logger.disable_logging()
    return {"status": "success", "logging_enabled": logger.logging_enabled}

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(path: str, request: Request):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Build the target URL
    target_url = f"{VLLM_BASE_URL}/{path}"

    # Get request body
    body = await request.body()

    # Check if this is a completions endpoint (only log these)
    is_completions_endpoint = "completions" in path

    # Check for streaming requests and reject them (only on completions endpoints)
    if is_completions_endpoint:
        try:
            if body:
                body_json = json.loads(body)
                if body_json.get("stream", False):
                    raise HTTPException(status_code=400, detail="Streaming requests are not supported by this proxy")
        except json.JSONDecodeError:
            pass

    # Forward headers (excluding host)
    headers = dict(request.headers)
    headers.pop("host", None)

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        try:
            # Handle non-streaming responses
            if is_completions_endpoint:
                # Log completions requests
                return await handle_non_streaming_request(
                    client, request_id, start_time, target_url, request.method, headers, body, should_log=True
                )
            else:
                # Don't log non-completions requests
                return await handle_non_streaming_request(
                    client, request_id, start_time, target_url, request.method, headers, body, should_log=False
                )

        except httpx.TimeoutException:
            # Client timeout - only log if completions endpoint
            if is_completions_endpoint:
                latency = time.time() - start_time
                logger.log_request(
                    request_id=request_id,
                    finish_reason="timeout",
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    latency_seconds=latency,
                )
            raise HTTPException(status_code=504, detail="Gateway Timeout")

        except Exception as e:
            # Other errors - only log if completions endpoint
            if is_completions_endpoint:
                latency = time.time() - start_time
                logger.log_request(
                    request_id=request_id,
                    finish_reason="error",
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    latency_seconds=latency,
                )
            raise

async def handle_non_streaming_request(
    client: httpx.AsyncClient,
    request_id: str,
    start_time: float,
    target_url: str,
    method: str,
    headers: dict,
    body: bytes,
    should_log: bool = True,
):
    # Forward the request
    response = await client.request(
        method=method,
        url=target_url,
        headers=headers,
        content=body,
    )

    latency = time.time() - start_time

    # Only log if requested (completions endpoints only)
    if should_log:
        # Parse response to extract token usage
        finish_reason = None
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        try:
            response_json = response.json()

            # Extract usage information (OpenAI API format)
            if "usage" in response_json:
                usage = response_json["usage"]
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")

            # Extract finish reason from first choice
            if "choices" in response_json and len(response_json["choices"]) > 0:
                finish_reason = response_json["choices"][0].get("finish_reason")

        except (json.JSONDecodeError, KeyError, IndexError):
            pass

        # Log the request
        logger.log_request(
            request_id=request_id,
            finish_reason=finish_reason or "unknown",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=latency,
        )

    # Return response to client
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers),
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
'''

    # Create wrapper script to set up environment and run proxy
    script = r'''
set -eu

echo "[*] Installing required packages..."
pip install --no-cache-dir fastapi uvicorn httpx

# Write the proxy server code
cat > /tmp/proxy_server.py << 'PROXY_EOF'
''' + proxy_script + r'''
PROXY_EOF

echo "[*] Starting vLLM proxy server..."
echo "[*] Proxying requests to: ${VLLM_SERVICE_URL}"
echo "[*] Logging to: ${LOGS_PATH}"

exec python /tmp/proxy_server.py
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
                        {"name": "LOGS_PATH", "value": str(logs_path / proxy_statistics_filename)},
                    ],
                    "volumeMounts": [
                        {"name": "artifacts", "mountPath": artifacts_pvc_mount_path},
                    ],
                }
            ],
            "volumes": [
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
            print(f"Deleted existing proxy pod {pod_name}; waiting for deletion...")
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
        raise TimeoutError(f"Timed out waiting for old proxy pod {pod_name} to delete")

    def create_or_patch_service() -> None:
        try:
            core.create_namespaced_service(namespace=namespace, body=service_manifest)
            print(f"Created proxy service {service_name}")
        except ApiException as e:
            if e.status == 409:
                core.patch_namespaced_service(
                    name=service_name,
                    namespace=namespace,
                    body=service_manifest,
                )
                print(f"Patched existing proxy service {service_name}")
            else:
                raise

    def create_pod() -> None:
        core.create_namespaced_pod(namespace=namespace, body=pod_manifest)
        print(f"Created proxy pod {pod_name}")

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

            print(f"Proxy pod phase={phase}, ready={ready}")

            if ready:
                return

            if phase in {"Failed", "Succeeded"}:
                raise RuntimeError(f"Proxy pod ended before becoming ready: phase={phase}")

            time.sleep(5)

        raise TimeoutError(f"Timed out waiting for proxy pod {pod_name} to become ready")

    create_or_patch_service()
    delete_pod_if_exists()
    create_pod()
    wait_for_pod_ready()

    print(f"vLLM proxy service URL: {service_url}")
    return service_url
