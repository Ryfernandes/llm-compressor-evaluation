from kfp import dsl, kubernetes
from mle_pipelines.components import (
    upload_to_huggingface,
    delete_local_model as delete_local_model_component,
)

PIPELINE_NAME = "huggingface-upload"


@dsl.pipeline(
    name=PIPELINE_NAME
)
def pipeline(
    session_id: str,
    model_name: str,
    delete_local_model: bool = False,
    models_pvc_name: str = "oneshot-pipeline-models-tier-2",
):
    """Pipeline to upload a compressed model from the models PVC to HuggingFace, with optional local model deletion."""

    upload_to_hf_task = upload_to_huggingface(
        session_id=session_id,
        model_name=model_name,
        models_mount_path="/models",
    )
    upload_to_hf_task.set_caching_options(enable_caching=False)
    upload_to_hf_task.set_memory_request("2Gi")
    kubernetes.mount_pvc(
        upload_to_hf_task,
        pvc_name=models_pvc_name,
        mount_path="/models",
    )
    kubernetes.use_secret_as_env(
        upload_to_hf_task,
        secret_name="ryan-test-hf-hub-secret",
        secret_key_to_env={"HF_WRITE": "HF_TOKEN"},
    )

    with dsl.Condition(delete_local_model == True):
        delete_local_model_task = (delete_local_model_component(
            session_id=session_id,
            model_name=model_name,
            models_mount_path="/models",
        ).after(upload_to_hf_task))
        delete_local_model_task.set_caching_options(enable_caching=False)
        delete_local_model_task.set_memory_request("512Mi")
        kubernetes.mount_pvc(
            delete_local_model_task,
            pvc_name=models_pvc_name,
            mount_path="/models",
        )


if __name__ == "__main__":
    from mle_pipelines.utils.generator import compile_pipeline
    compile_pipeline(pipeline, PIPELINE_NAME)
