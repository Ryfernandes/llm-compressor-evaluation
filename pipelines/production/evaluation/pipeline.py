from kfp import dsl, kubernetes
from components import (
    cleanup_model_staging,
    collate_results,
    create_vllm_server,
    delete_vllm_server,
    test_vllm_server,
    upload_to_github,
    validate_session_id,
    evaluate_model,
)

PIPELINE_NAME = "llm-evaluation"

@dsl.pipeline(
    name=PIPELINE_NAME
)
def pipeline(
    # Model spec
    model_id: str = "Qwen/Qwen3-8B",
    reasoning_parser: str = "",
    # Session spec
    session_id: str = "evaluation-pipeline-artifacts-tier-2",
    # Evaluation spec
    evaluation_config_path: str = "gsm8k_platinum_cot_llama",
    # PVC spec
    artifacts_pvc_name: str = "evaluation-pipeline-artifacts-tier-2",
    evaluation_config_pvc_name: str = "evaluation-pipeline-evaluation-configs-tier-2",
    model_server_pvc_name: str = "evaluation-pipeline-model-server-tier-2",
):
    """Pipeline to evaluate a model from HuggingFace using vLLM, lm-eval and lighteval."""

    # Validate that the session_id has not already been used in the artifacts PVC.
    # If it has not been used, make the session directory to "claim" the session_id.
    validate_session_id_task = validate_session_id(session_id=session_id, mount_path="/artifacts")
    kubernetes.mount_pvc(
        validate_session_id_task,
        pvc_name=artifacts_pvc_name,
        mount_path="/artifacts"
    )

    ### Baseline model flow
    create_vllm_task = create_vllm_server(session_id=session_id_task.output, reasoning_parser=reasoning_parser, pod_suffix="b", model=model_id)
    create_vllm_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        create_vllm_task,
        pvc_name="evaluation-pipeline-model-server-tier-2",
        mount_path="/tier2"
    )
    
    test_vllm_task = test_vllm_server(service_url=create_vllm_task.output, model=model_id)
    test_vllm_task.set_caching_options(enable_caching=False)

    evaluation_task = (
        evaluate_model(session_id=session_id_task.output, service_url=test_vllm_task.output, tasks=evaluation_tasks, reasoning_parser=reasoning_parser, save_prefix="baseline", model_path=model_id)
        .set_accelerator_type("nvidia.com/gpu")
        .set_accelerator_limit("1")
    )
    evaluation_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        evaluation_task,
        pvc_name=pvc_name,
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

    ### Collate results from both flows
    collate_task = (
        collate_results(session_id=session_id_task.output, model_id=model_id, compression_recipe=compression_recipe)
        .after(delete_vllm_task)
    )
    collate_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        collate_task,
        pvc_name=pvc_name,
        mount_path="/tier2"
    )

    ### Upload collated results to GitHub
    github_upload_task = upload_to_github(session_id=session_id_task.output).after(collate_task)
    github_upload_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        github_upload_task,
        pvc_name=pvc_name,
        mount_path="/tier2"
    )
    kubernetes.use_secret_as_env(
        github_upload_task,
        secret_name="evaluation-pipeline-results-gh",
        secret_key_to_env={"GITHUB_TOKEN": "GITHUB_TOKEN"}
    )