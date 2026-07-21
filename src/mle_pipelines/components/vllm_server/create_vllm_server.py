from mle_pipelines.utils.constants import (
    get_hardware_constants,
    get_openshift_constants,
    get_evaluation_allow_ignore,
)
from mle_pipelines.utils.models.model_config import load_model_config
from .utils import (
    get_vllm_server_script,
    get_vllm_pod_manifest,
    get_proxy_server_script,
    get_proxy_pod_manifest,
    get_s3_upload_sidecar_spec,
    get_svc_manifest,
    send_test_request,
    delete_pod_if_exists,
    create_or_patch_service,
    create_pod,
    wait_for_pod_ready,
    wait_for_service_healthy,
)

from kfp import dsl

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["kubernetes", "requests", "huggingface-hub", "pydantic"],
    additional_funcs=[
        get_hardware_constants, 
        get_openshift_constants, 
        get_evaluation_allow_ignore, 
        load_model_config,
        get_vllm_server_script,
        get_vllm_pod_manifest,
        get_proxy_server_script,
        get_proxy_pod_manifest,
        get_s3_upload_sidecar_spec,
        get_svc_manifest,
        send_test_request,
        delete_pod_if_exists,
        create_or_patch_service,
        create_pod,
        wait_for_pod_ready,
        wait_for_service_healthy,
    ]
)
def create_vllm_server(
    model: str,
    model_config: dict,
    vllm_image: str = "vllm/vllm-openai:latest",
    with_proxy: bool = True,
) -> dict:
    """
    expects hf_token env
    expects hf model tag or input artifact
    with proxy to configure if poxy on or off
    explain the output format

    The model serving script might be completely broken with artifact models and/or huggingface ignores
    Make sure to revisit and get this right
    """
    import os
    import math
    from kubernetes import client, config
    from huggingface_hub import snapshot_download
    
    config.load_incluster_config()
    core = client.CoreV1Api()

    print("Loding constants and preparing model")

    # Get shared constants
    hardware_constants = get_hardware_constants()
    openshift_constants = get_openshift_constants()

    # Assign hardware constants
    H100_VRAM_GI = hardware_constants.get("H100_VRAM_GI")
    EVALUATION_KV_CACHE_VRAM_BUFFER_GI = hardware_constants.get("EVALUATION_KV_CACHE_VRAM_BUFFER_GI")
    EVALUATION_MODEL_STORAGE_BUFFER_GI = hardware_constants.get("EVALUATION_MODEL_STORAGE_BUFFER_GI")

    # Assign OpenShift constants
    NAMESPACE = openshift_constants.get("NAMESPACE")
    SERVICE_ACCOUNT_NAME = openshift_constants.get("SERVICE_ACCOUNT_NAME")
    HF_SECRET_NAME = openshift_constants.get("HF_SECRET_NAME")
    H100_SELECTOR_KEY = openshift_constants.get("H100_SELECTOR_KEY")
    TIER1_STORAGE_CLASS = openshift_constants.get("TIER1_STORAGE_CLASS")

    # Validate the model config with Pydantic
    validated_model_config = load_model_config(model_config)

    print("Model validated successfully")

    # Assign allow/ignore pattern constants
    ALLOW_PATTERNS, IGNORE_PATTERNS = get_evaluation_allow_ignore()

    RUN_ID = '{{workflow.uid}}'

    print(f"Run id loaded with value {RUN_ID}")

    # Get the size of the model, in order to determine the number of GPUs required
    if isinstance(model, str):
        # HF model id is provided, do a dry run download to determine size
        dry_run_files = snapshot_download(
            repo_id=model,
            token=os.environ.get("HF_TOKEN"),
            dry_run=True,
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=["original/**/*"] + IGNORE_PATTERNS,
        )

        model_size_bytes = sum(
            f.file_size for f in dry_run_files
            if f.file_size is not None
        )
        serveable_model = model
    else:
        # Local model is provided, determine size of the directory
        import fnmatch

        model_size_bytes = 0
        for root, dirs, files in os.walk(model.path):
            if "original" in dirs:
                dirs.remove("original")
            for f in files:
                if any(fnmatch.fnmatch(f, p) for p in IGNORE_PATTERNS):
                    continue
                if any(fnmatch.fnmatch(f, p) for p in ALLOW_PATTERNS):
                    model_size_bytes += os.path.getsize(os.path.join(root, f))

        serveable_model = model.path
    
    print("Got model size")
    
    model_size_gi = model_size_bytes / (1024 ** 3)
    inference_size_gi = model_size_gi + EVALUATION_KV_CACHE_VRAM_BUFFER_GI
    
    # Tune tp, if it is smaller than required
    min_tp = math.ceil(inference_size_gi / H100_VRAM_GI)

    if validated_model_config.tp < min_tp:
        adjusted_tp = 2 ** math.ceil(math.log2(min_tp))
        validated_model_config.tp = adjusted_tp
    
    # Assign the number of GPUs required for the vLLM server pod
    gpus = validated_model_config.tp * validated_model_config.dp

    is_local_model = not isinstance(model, str)

    # Compute tier1 ephemeral volume size for HF model downloads
    tier1_size_gi = math.ceil(model_size_gi) + EVALUATION_MODEL_STORAGE_BUFFER_GI

    # --------------------------------------
    # Create the vLLM server pod and service
    # --------------------------------------
    server_pod_name = f"pipeline-vllm-server-pod-{RUN_ID}"
    server_svc_name = f"pipeline-vllm-server-svc-{RUN_ID}"

    labels = {
        "app.kubernetes.io/instance": server_pod_name,
    }

    service_url = f"http://{server_svc_name}.{NAMESPACE}.svc.cluster.local:8000"

    print("Retrieving server pod info")

    script = get_vllm_server_script()
    server_sidecar = get_s3_upload_sidecar_spec(
        log_file_path="/logs/vllm_serve.log",
        s3_key=f"pipeline-runs/{RUN_ID}/logs/vllm_serve.log",
    )
    vllm_pod_manifest = get_vllm_pod_manifest(
        server_pod_name,
        labels,
        NAMESPACE,
        SERVICE_ACCOUNT_NAME,
        HF_SECRET_NAME,
        H100_SELECTOR_KEY,
        vllm_image,
        script,
        gpus,
        serveable_model,
        validated_model_config,
        is_local_model,
        TIER1_STORAGE_CLASS,
        tier1_size_gi,
        server_sidecar,
    )
    vllm_svc_manifest = get_svc_manifest(
        server_svc_name,
        NAMESPACE,
        labels,
    )

    print("Creating server pod")

    create_or_patch_service(
        server_svc_name,
        NAMESPACE,
        vllm_svc_manifest,
        core,
    )
    delete_pod_if_exists(
        server_pod_name,
        NAMESPACE,
        core,
    )
    create_pod(
        server_pod_name,
        NAMESPACE,
        vllm_pod_manifest,
        core,
    )
    wait_for_pod_ready(
        server_pod_name,
        NAMESPACE,
        core,
    )
    wait_for_service_healthy(
        service_url,
    )

    # Here, extract the startup information from the vLLM server pod logs

    # ------------------------------------------------------
    # Conditionally, create the proxy server pod and service
    # ------------------------------------------------------
    if with_proxy:
        print("Retrieving proxy pod info")

        proxy_pod_name = f"pipeline-vllm-proxy-pod-{RUN_ID}"
        proxy_svc_name = f"pipeline-vllm-proxy-svc-{RUN_ID}"

        proxy_labels = {
            "app.kubernetes.io/instance": proxy_pod_name,
        }

        proxy_service_url = f"http://{proxy_svc_name}.{NAMESPACE}.svc.cluster.local:8000"

        proxy_script = get_proxy_server_script()
        proxy_sidecar = get_s3_upload_sidecar_spec(
            log_file_path="/logs/vllm_requests.jsonl",
            s3_key=f"pipeline-runs/{RUN_ID}/logs/vllm_requests.jsonl",
        )
        proxy_pod_manifest = get_proxy_pod_manifest(
            proxy_pod_name,
            proxy_labels,
            NAMESPACE,
            SERVICE_ACCOUNT_NAME,
            proxy_script,
            service_url,
            RUN_ID,
            proxy_sidecar,
        )
        proxy_svc_manifest = get_svc_manifest(
            proxy_svc_name,
            NAMESPACE,
            proxy_labels,
        )

        print("Creating proxy pod")

        create_or_patch_service(
            proxy_svc_name,
            NAMESPACE,
            proxy_svc_manifest,
            core,
        )
        delete_pod_if_exists(
            proxy_pod_name,
            NAMESPACE,
            core,
        )
        create_pod(
            proxy_pod_name,
            NAMESPACE,
            proxy_pod_manifest,
            core,
        )
        wait_for_pod_ready(
            proxy_pod_name,
            NAMESPACE,
            core,
        )

        print("Testing proxy server")
        test_result = send_test_request(proxy_service_url, serveable_model)
        return {
            "service_url": proxy_service_url,
            "vllm_service_url": service_url,
            "test_result": test_result,
        }

    print("Testing server")
    test_result = send_test_request(service_url, serveable_model)
    return {
        "service_url": service_url,
        "test_result": test_result,
    }