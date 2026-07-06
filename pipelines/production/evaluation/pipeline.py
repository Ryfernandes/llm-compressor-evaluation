from kfp import dsl, kubernetes
from components import (
    cleanup_model_staging,
    collate_results,
    create_vllm_server,
    create_vllm_proxy,
    delete_vllm_server,
    test_vllm_server,
    test_vllm_proxy,
    upload_to_github,
    validate_session_id,
    validate_config,
    lm_eval_evaluation,
    lighteval_evaluation,
    save_startup_statistics,
)

PIPELINE_NAME = "llm-evaluation"

@dsl.pipeline(
    name=PIPELINE_NAME
)
def pipeline(
    # Model spec
    model_id: str,
    # Session spec
    session_id: str,
    # Evaluation spec
    config_filename: str,
    # PVC spec
    artifacts_pvc_name: str = "evaluation-pipeline-artifacts-tier-2",
    configs_pvc_name: str = "evaluation-pipeline-configs-tier-2",
    model_server_pvc_name: str = "evaluation-pipeline-model-server-tier-2",
    packages_pvc_name: str = "evaluation-pipeline-packages-tier-2",
):
    """Pipeline to evaluate a model from HuggingFace using vLLM, lm-eval and lighteval."""

    # Validate that the session_id has not already been used in the artifacts PVC.
    # If it has not been used, make the session directory to "claim" the session_id.
    validate_session_id_task = validate_session_id(session_id=session_id, mount_path="/artifacts")
    validate_session_id_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        validate_session_id_task,
        pvc_name=artifacts_pvc_name,
        mount_path="/artifacts"
    )

    # Validate config file structure and required fields
    validate_config_task = (validate_config(
        config_filename=config_filename,
        configs_pvc_mount_path="/configs"
    ).after(validate_session_id_task))
    validate_config_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        validate_config_task,
        pvc_name=configs_pvc_name,
        mount_path="/configs"
    )

    # Create cleanup task that will take down any vLLM server, PVCs, and services created for this session_id
    cleanup_task = delete_vllm_server(session_id=session_id)
    cleanup_task.set_caching_options(enable_caching=False)

    # Run the cleanup task as soon as the pipeline exits, for any reason
    with dsl.ExitHandler(cleanup_task):
        # Create vLLM server pod separate from the pipeline to serve the model for evaluation.
        # The task waits for the server to become ready and exposes it as a Kubernetes service
        create_vllm_task = (create_vllm_server(
            model=model_id,
            session_id=session_id,
            config_filename=config_filename,
            model_server_pvc_name=model_server_pvc_name,
            configs_pvc_mount_path="/configs",
            artifacts_pvc_name=artifacts_pvc_name,
            artifacts_pvc_mount_path="/artifacts"
        ).after(validate_config_task))
        create_vllm_task.set_caching_options(enable_caching=False)
        kubernetes.mount_pvc(
            create_vllm_task,
            pvc_name=model_server_pvc_name,
            mount_path="/server"
        )
        kubernetes.mount_pvc(
            create_vllm_task,
            pvc_name=configs_pvc_name,
            mount_path="/configs"
        )
        kubernetes.mount_pvc(
            create_vllm_task,
            pvc_name=artifacts_pvc_name,
            mount_path="/artifacts"
        )
        kubernetes.use_secret_as_env(
            create_vllm_task,
            secret_name="ryan-test-hf-hub-secret",
            secret_key_to_env={"HF_TOKEN": "HF_TOKEN"}
        )

        test_vllm_task = test_vllm_server(service_url=create_vllm_task.output, model=model_id)
        test_vllm_task.set_caching_options(enable_caching=False)

        # Create proxy between evaluation harness and vLLM server for request/response logging
        create_proxy_task = (create_vllm_proxy(
            session_id=session_id,
            vllm_service_url=create_vllm_task.output,
            artifacts_pvc_name=artifacts_pvc_name,
            artifacts_pvc_mount_path="/artifacts"
        ).after(test_vllm_task))
        create_proxy_task.set_caching_options(enable_caching=False)
        kubernetes.mount_pvc(
            create_proxy_task,
            pvc_name=artifacts_pvc_name,
            mount_path="/artifacts"
        )

        # Test proxy to ensure it can forward requests to vLLM server
        test_proxy_task = test_vllm_proxy(proxy_url=create_proxy_task.output, model=model_id)
        test_proxy_task.set_caching_options(enable_caching=False)

        # Save vLLM startup statistics from server logs
        save_stats_task = (
            save_startup_statistics(
                session_id=session_id,
                config_filename=config_filename,
                artifacts_pvc_mount_path="/artifacts",
                configs_pvc_mount_path="/configs"
            )
            .after(test_proxy_task)
        )
        save_stats_task.set_caching_options(enable_caching=False)
        kubernetes.mount_pvc(
            save_stats_task,
            pvc_name=artifacts_pvc_name,
            mount_path="/artifacts"
        )
        kubernetes.mount_pvc(
            save_stats_task,
            pvc_name=configs_pvc_name,
            mount_path="/configs"
        )

        # Run evaluation using proxy URL (routes through proxy for logging)
        lm_evaluation_task = (
            lm_eval_evaluation(
                service_url=create_proxy_task.output,
                config_filename=config_filename,
                session_id=session_id,
                model_path=model_id,
                artifacts_pvc_mount_path="/artifacts",
                configs_pvc_mount_path="/configs",
                packages_pvc_mount_path="/packages"
            )
            .after(save_stats_task)
            .set_accelerator_type("nvidia.com/gpu")
            .set_accelerator_limit("1")
        )
        lm_evaluation_task.set_caching_options(enable_caching=False)
        kubernetes.mount_pvc(
            lm_evaluation_task,
            pvc_name=artifacts_pvc_name,
            mount_path="/artifacts"
        )
        kubernetes.mount_pvc(
            lm_evaluation_task,
            pvc_name=configs_pvc_name,
            mount_path="/configs"
        )
        kubernetes.mount_pvc(
            lm_evaluation_task,
            pvc_name=packages_pvc_name,
            mount_path="/packages"
        )
        kubernetes.use_secret_as_env(
            lm_evaluation_task,
            secret_name="ryan-test-hf-hub-secret",
            secret_key_to_env={"HF_TOKEN": "HF_TOKEN"}
        )

        # Run lighteval evaluation using proxy URL (routes through proxy for logging)
        lighteval_evaluation_task = (
            lighteval_evaluation(
                service_url=create_proxy_task.output,
                config_filename=config_filename,
                session_id=session_id,
                model_path=model_id,
                artifacts_pvc_mount_path="/artifacts",
                configs_pvc_mount_path="/configs",
                packages_pvc_mount_path="/packages"
            )
            .after(lm_evaluation_task)
            .set_accelerator_type("nvidia.com/gpu")
            .set_accelerator_limit("1")
        )
        lighteval_evaluation_task.set_caching_options(enable_caching=False)
        kubernetes.mount_pvc(
            lighteval_evaluation_task,
            pvc_name=artifacts_pvc_name,
            mount_path="/artifacts"
        )
        kubernetes.mount_pvc(
            lighteval_evaluation_task,
            pvc_name=configs_pvc_name,
            mount_path="/configs"
        )
        kubernetes.mount_pvc(
            lighteval_evaluation_task,
            pvc_name=packages_pvc_name,
            mount_path="/packages"
        )
        kubernetes.use_secret_as_env(
            lighteval_evaluation_task,
            secret_name="ryan-test-hf-hub-secret",
            secret_key_to_env={"HF_TOKEN": "HF_TOKEN"}
        )

        # Collate results from evaluation runs (both lm_eval and lighteval)
        collate_task = (
            collate_results(
                session_id=session_id,
                model_id=model_id,
                config_filename=config_filename,
                artifacts_pvc_mount_path="/artifacts",
                configs_pvc_mount_path="/configs"
            )
            .after(lighteval_evaluation_task)
        )
        collate_task.set_caching_options(enable_caching=False)
        kubernetes.mount_pvc(
            collate_task,
            pvc_name=artifacts_pvc_name,
            mount_path="/artifacts"
        )
        kubernetes.mount_pvc(
            collate_task,
            pvc_name=configs_pvc_name,
            mount_path="/configs"
        )

        """
        ### Upload collated results to GitHub
        github_upload_task = upload_to_github(session_id=session_id).after(collate_task)
        github_upload_task.set_caching_options(enable_caching=False)
        kubernetes.mount_pvc(
            github_upload_task,
            pvc_name=artifacts_pvc_name,
            mount_path="/tier2"
        )
        kubernetes.use_secret_as_env(
            github_upload_task,
            secret_name="evaluation-pipeline-results-gh",
            secret_key_to_env={"GITHUB_TOKEN": "GITHUB_TOKEN"}
        )
        """
