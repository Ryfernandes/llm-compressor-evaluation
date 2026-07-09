from kfp import dsl


@dsl.component(
    base_image="python:3.12",
)
def clean_up_compressed_model(
    save_model_locally: bool,
    session_id: str,
    model_name: str,
    models_mount_path: str = "/models",
) -> None:
    """
    Delete the compressed model from the PVC if save_model_locally is False.
    Only runs cleanup when the model is not meant to be kept locally.
    """

    if save_model_locally:
        print("Keeping compressed model locally (save_model_locally=True)")
        return

    import shutil
    from pathlib import Path

    model_path = Path(models_mount_path) / "sessions" / session_id / model_name

    if model_path.exists():
        print(f"Cleaning up compressed model: {model_path}")
        shutil.rmtree(model_path)
        print("Compressed model removed")
    else:
        print(f"Model directory does not exist (already cleaned up): {model_path}")
