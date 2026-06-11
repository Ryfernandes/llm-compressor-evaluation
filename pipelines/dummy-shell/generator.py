# Simple, dummy pipeline to demonstrate KFP functionality and verify pipeline execution

from typing import List, Optional

from kfp import dsl, compiler

PIPELINE_NAME = "dummy_shell"

@dsl.component(
    base_image="quay.io/rh-ee-ryfernan/dummy-llmcompressor-image:0.1"
)
def write_file (message: str, output_path: dsl.Output[dsl.Artifact]) -> None:
    # Call shell with message
    import subprocess
    subprocess.run(
        ["bash", "artifact-generator.sh", message, output_path.path],
        check=True, # Python step will fail if the command fails
    )

@dsl.component
def read_file (input_path: dsl.Input[dsl.Artifact]) -> str:
    with open(input_path.path, "r") as f:
        message = f.read()
    return f"Received message: {message}"

@dsl.pipeline
def shell_pipeline(message: str = "Hello, world!") -> str:
    write_task = write_file(message=message)
    read_task = read_file(input_path=write_task.outputs["output_path"])
    return read_task.output

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=shell_pipeline, package_path=f"{PIPELINE_NAME}.yaml"
    )