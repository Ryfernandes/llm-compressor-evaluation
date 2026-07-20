from kfp import dsl, kubernetes
from mle_pipelines.components import (
    validate_yaml_config,
    create_session_id,
    get_model_size,
    validate_compression_session_id as validate_session_id,
    process_dataset,
    compress_model_oneshot as compress_model,
    clean_up_dataset,
)

PIPELINE_NAME = "llm-compression-oneshot"


@dsl.pipeline(
    name=PIPELINE_NAME
)
def pipeline(
    # Model spec
    model_id: str,
    # Compression spec
    yaml_filename: str,
    model_name: str,
    # Dataset spec
    dataset_id: str = "",
    dataset_split: str = "",
    num_calibration_samples: int = 256,
    max_sequence_length: int = 512,
    # PVC spec
    models_pvc_name: str = "oneshot-pipeline-models-tier-2",
    yaml_pvc_name: str = "oneshot-pipeline-yamls-tier-2",
    datasets_pvc_name: str = "oneshot-pipeline-datasets-tier-2",
    packages_pvc_name: str = "oneshot-pipeline-packages-tier-2",
):
    """Pipeline to compress a model from HuggingFace using LLM Compressor."""

    # Validate that the YAML recipe file exists and is syntactically valid
    validate_yaml_config_task = validate_yaml_config(
        yaml_filename=yaml_filename,
        yamls_mount_path="/yamls"
    )
    validate_yaml_config_task.set_caching_options(enable_caching=False)
    validate_yaml_config_task.set_memory_request("512Mi")
    kubernetes.mount_pvc(
        validate_yaml_config_task,
        pvc_name=yaml_pvc_name,
        mount_path="/yamls"
    )

    # Generate a unique session ID
    create_session_id_task = create_session_id()
    create_session_id_task.set_caching_options(enable_caching=False)
    create_session_id_task.set_memory_request("256Mi")

    # Determine model size and compute memory request for compression
    get_model_size_task = (get_model_size(
        model_id=model_id,
    ).after(validate_yaml_config_task))
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
    ).after(validate_yaml_config_task))
    validate_session_id_task.set_caching_options(enable_caching=False)
    validate_session_id_task.set_memory_request("512Mi")
    kubernetes.mount_pvc(
        validate_session_id_task,
        pvc_name=models_pvc_name,
        mount_path="/models"
    )

    # Load, preprocess, and save the calibration dataset
    process_dataset_task = (process_dataset(
        model_id=model_id,
        dataset_id=dataset_id,
        dataset_split=dataset_split,
        num_calibration_samples=num_calibration_samples,
        max_sequence_length=max_sequence_length,
        session_id=create_session_id_task.output,
        models_mount_path="/models",
        datasets_mount_path="/datasets"
    ).after(validate_session_id_task))
    process_dataset_task.set_caching_options(enable_caching=False)
    process_dataset_task.set_memory_request("4Gi")
    kubernetes.mount_pvc(
        process_dataset_task,
        pvc_name=models_pvc_name,
        mount_path="/models"
    )
    kubernetes.mount_pvc(
        process_dataset_task,
        pvc_name=datasets_pvc_name,
        mount_path="/datasets"
    )
    kubernetes.use_secret_as_env(
        process_dataset_task,
        secret_name="ryan-test-hf-hub-secret",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"}
    )

    # Run oneshot compression with the YAML recipe and calibration dataset
    compress_model_task = (
        compress_model(
            model_id=model_id,
            yaml_filename=yaml_filename,
            session_id=create_session_id_task.output,
            model_name=model_name,
            num_calibration_samples=num_calibration_samples,
            max_sequence_length=max_sequence_length,
            models_mount_path="/models",
            yamls_mount_path="/yamls"
        )
        .after(process_dataset_task)
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
    kubernetes.mount_pvc(
        compress_model_task,
        pvc_name=yaml_pvc_name,
        mount_path="/yamls"
    )
    kubernetes.use_secret_as_env(
        compress_model_task,
        secret_name="ryan-test-hf-hub-secret",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"}
    )

    # Clean up the calibration dataset after compression
    clean_up_dataset_task = (clean_up_dataset(
        session_id=create_session_id_task.output,
        models_mount_path="/models"
    ).after(compress_model_task))
    clean_up_dataset_task.set_caching_options(enable_caching=False)
    clean_up_dataset_task.set_memory_request("512Mi")
    kubernetes.mount_pvc(
        clean_up_dataset_task,
        pvc_name=models_pvc_name,
        mount_path="/models"
    )


if __name__ == "__main__":
    from mle_pipelines.utils.generator import compile_pipeline
    compile_pipeline(pipeline, PIPELINE_NAME)
