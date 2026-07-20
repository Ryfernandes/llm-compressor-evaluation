from kfp import dsl, kubernetes
from mle_pipelines.components import (
    validate_scheme,
    create_session_id,
    get_model_size,
    validate_compression_session_id as validate_session_id,
    compress_model_free_ptq as compress_model,
)

PIPELINE_NAME = "llm-compression-model-free-ptq"


@dsl.pipeline(
    name=PIPELINE_NAME
)
def pipeline(
    # Model spec
    model_id: str,
    # Compression spec
    scheme: str,
    model_name: str,
    ignore: str = "",
    max_workers: int = 15,
    # PVC spec
    models_pvc_name: str = "model-free-ptq-pipeline-models-tier-2",
):
    """Pipeline to compress a model from HuggingFace using LLM Compressor's model-free PTQ."""

    # Validate that the scheme resolves to a known quantization preset
    validate_scheme_task = validate_scheme(
        scheme=scheme,
    )
    validate_scheme_task.set_caching_options(enable_caching=False)
    validate_scheme_task.set_memory_request("512Mi")

    # Generate a unique session ID
    create_session_id_task = create_session_id()
    create_session_id_task.set_caching_options(enable_caching=False)
    create_session_id_task.set_memory_request("256Mi")

    # Determine model size and compute memory request for compression
    get_model_size_task = (get_model_size(
        model_id=model_id,
    ).after(validate_scheme_task))
    get_model_size_task.set_caching_options(enable_caching=False)
    get_model_size_task.set_memory_request("1Gi")
    kubernetes.use_secret_as_env(
        get_model_size_task,
        secret_name="ryan-test-hf-hub-secret",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"}
    )

    # Validate session ID uniqueness and claim the session directory
    validate_session_id_task = (validate_session_id(
        session_id=create_session_id_task.output,
        models_mount_path="/models"
    ).after(validate_scheme_task))
    validate_session_id_task.set_caching_options(enable_caching=False)
    validate_session_id_task.set_memory_request("512Mi")
    kubernetes.mount_pvc(
        validate_session_id_task,
        pvc_name=models_pvc_name,
        mount_path="/models"
    )

    # Run model-free PTQ compression with the specified scheme
    compress_model_task = (
        compress_model(
            model_id=model_id,
            scheme=scheme,
            session_id=create_session_id_task.output,
            model_name=model_name,
            ignore=ignore,
            max_workers=max_workers,
            models_mount_path="/models",
        )
        .after(validate_session_id_task)
        .after(get_model_size_task)
        .set_accelerator_type("nvidia.com/gpu")
        .set_accelerator_limit("1")
    )
    compress_model_task.set_caching_options(enable_caching=False)
    compress_model_task.set_memory_request(get_model_size_task.output)
    kubernetes.add_node_selector(
        compress_model_task,
        label_key="node-role.kubernetes.io/up-h100mcp",
        label_value=""
    )
    kubernetes.mount_pvc(
        compress_model_task,
        pvc_name=models_pvc_name,
        mount_path="/models"
    )
    kubernetes.use_secret_as_env(
        compress_model_task,
        secret_name="ryan-test-hf-hub-secret",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"}
    )


if __name__ == "__main__":
    from mle_pipelines.utils.generator import compile_pipeline
    compile_pipeline(pipeline, PIPELINE_NAME)
