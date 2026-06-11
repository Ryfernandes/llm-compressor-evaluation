from kfp import dsl, compiler, kubernetes

PIPELINE_NAME = "pvc-access-test"

@dsl.component
def run_pvc_script() -> str:
    import importlib.util
    from pathlib import Path
    file_path=Path("/tier1/code/hello_world.py")

    spec = importlib.util.spec_from_file_location("hello_world", file_path)
    hello_world = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hello_world)
    
    hello_world.say_hello()

    return "Test completed, check component output"

@dsl.pipeline
def pvc_pipeline() -> None:
    pvc_task = (
        run_pvc_script()
        .set_accelerator_type("nvidia.com/gpu")
        .set_accelerator_limit("1")
    )

    kubernetes.mount_pvc(
        pvc_task,
        pvc_name="ryan-interactive-tier-1",
        mount_path="/tier1"
    )

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pvc_pipeline, package_path=f"{PIPELINE_NAME}.yaml"
    )