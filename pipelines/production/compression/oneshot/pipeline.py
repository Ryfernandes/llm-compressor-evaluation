from kfp import dsl, kubernetes
from components import (
    validate_yaml_config,
    create_session_id,
    validate_session_id,
    process_dataset,
    compress_model,
    upload_to_huggingface,
    clean_up_dataset,
    clean_up_compressed_model,
)

PIPELINE_NAME = "llm-compression-oneshot"


@dsl.pipeline(
    name=PIPELINE_NAME
)
def pipeline(
    # Model spec
    model_id: str,
    # Dataset spec
    dataset_id: str,
    dataset_split: str,
    num_calibration_samples: int,
    max_sequence_length: int,
    # Compression spec
    yaml_filename: str,
    # Upload spec
    upload_to_hf: bool = False,
    save_model_locally: bool = True,
    # PVC spec
    models_pvc_name: str = "",
    yaml_pvc_name: str = "",
    packages_pvc_name: str = "",
):
    """Pipeline to compress a model from HuggingFace using LLM Compressor, optionally uploading it to HuggingFace after compression."""

    # Validate that the YAML recipe file exists and is syntactically valid
    validate_yaml_config_task = validate_yaml_config(
        yaml_filename=yaml_filename,
        yamls_mount_path="/yamls"
    )
    validate_yaml_config_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        validate_yaml_config_task,
        pvc_name=yaml_pvc_name,
        mount_path="/yamls"
    )

    # Generate a unique session ID
    create_session_id_task = create_session_id()
    create_session_id_task.set_caching_options(enable_caching=False)

    # Validate session ID uniqueness and claim the session directory
    validate_session_id_task = (validate_session_id(
        session_id=create_session_id_task.output,
        models_mount_path="/models"
    ).after(validate_yaml_config_task))
    validate_session_id_task.set_caching_options(enable_caching=False)
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
        models_mount_path="/models"
    ).after(validate_session_id_task))
    process_dataset_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        process_dataset_task,
        pvc_name=models_pvc_name,
        mount_path="/models"
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
            num_calibration_samples=num_calibration_samples,
            max_sequence_length=max_sequence_length,
            models_mount_path="/models",
            yamls_mount_path="/yamls"
        )
        .after(process_dataset_task)
        .set_accelerator_type("nvidia.com/gpu")
        .set_accelerator_limit("1")
    )
    compress_model_task.set_caching_options(enable_caching=False)
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
    kubernetes.mount_pvc(
        clean_up_dataset_task,
        pvc_name=models_pvc_name,
        mount_path="/models"
    )

    # Upload compressed model to HuggingFace (conditional on upload_to_hf)
    upload_to_hf_task = (upload_to_huggingface(
        upload_to_hf=upload_to_hf,
        session_id=create_session_id_task.output,
        model_name=compress_model_task.output,
        models_mount_path="/models"
    ).after(compress_model_task))
    upload_to_hf_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        upload_to_hf_task,
        pvc_name=models_pvc_name,
        mount_path="/models"
    )
    kubernetes.use_secret_as_env(
        upload_to_hf_task,
        secret_name="ryan-test-hf-hub-secret",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"}
    )

    # Clean up compressed model if not saving locally (after upload completes)
    clean_up_model_task = (clean_up_compressed_model(
        save_model_locally=save_model_locally,
        session_id=create_session_id_task.output,
        model_name=compress_model_task.output,
        models_mount_path="/models"
    ).after(upload_to_hf_task))
    clean_up_model_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        clean_up_model_task,
        pvc_name=models_pvc_name,
        mount_path="/models"
    )
