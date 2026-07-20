from kfp import dsl

@dsl.component(
    base_image="python:3.12",
)
def delete_local_model(
    session_id: str,
    model_name: str,
    models_mount_path: str = "/models",
) -> None:
    """
    Delete the compressed model from the PVC.
    """

    import shutil
    from pathlib import Path

    model_path = Path(models_mount_path) / "sessions" / session_id / model_name

    if model_path.exists():
        print(f"Deleting local model: {model_path}")
        shutil.rmtree(model_path)
        print("Local model deleted")
    else:
        print(f"Model directory does not exist (already deleted): {model_path}")
