# Simple, dummy pipeline to demonstrate KFP functionality and verify pipeline execution

from typing import List, Optional

from kfp import dsl, compiler

PIPELINE_NAME = "gpu_pipeline"

@dsl.component(
    base_image="quay.io/rh-ee-ryfernan/dummy-llmcompressor-image:0.1"
)
def test_gpu_access(message: str) -> str:
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
   
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")

    return "GPU access test completed"
      

@dsl.pipeline
def gpu_pipeline(message: str = "Hello, world!") -> str:
    gpu_task = (
        test_gpu_access(message=message)
        .set_accelerator_type("nvidia.com/gpu")
        .set_accelerator_limit("1")
    )
    return gpu_task.output

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=gpu_pipeline, package_path=f"{PIPELINE_NAME}.yaml"
    )