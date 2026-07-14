from kfp import dsl


@dsl.component(base_image="python:3.12")
def validate_model(
    local_model: bool,
    model_id: str,
    local_model_path: str,
    local_models_mount_path: str = "/local-models",
) -> str:
    """Validate model source parameters and return the served model name."""
    from pathlib import Path

    if local_model:
        if not local_model_path:
            raise ValueError(
                "local_model_path must be set when local_model is True"
            )

        full_path = Path(local_models_mount_path) / local_model_path
        if not full_path.exists():
            raise FileNotFoundError(
                f"Local model path does not exist: {full_path}"
            )
        if not full_path.is_dir():
            raise ValueError(
                f"Local model path is not a directory: {full_path}"
            )

        print(f"Validated local model at {full_path}")
        return local_model_path
    else:
        if not model_id:
            raise ValueError(
                "model_id must be set when local_model is False"
            )

        print(f"Validated remote model: {model_id}")
        return model_id
