from kfp import dsl, kubernetes
from mle_pipelines.components.vllm_server.create_vllm_server import create_vllm_server

PIPELINE_NAME = "test-vllm-server"

@dsl.pipeline(
    name=PIPELINE_NAME
)
def pipeline(
    model: str,
    model_config: dict,
    vllm_image: str = "vllm/vllm-openai:latest",
    with_proxy: bool = True,
):
    create_vllm_task = create_vllm_server(
        model=model,
        model_config=model_config,
        vllm_image=vllm_image,
        with_proxy=with_proxy,
    )
    create_vllm_task.set_caching_options(enable_caching=False)
    kubernetes.use_secret_as_env(
        create_vllm_task,
        secret_name="ryan-test-hf-hub-secret",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
    )


if __name__ == "__main__":
    from mle_pipelines.utils.generator import compile_pipeline
    compile_pipeline(pipeline, PIPELINE_NAME)
