from kfp import dsl


@dsl.component(
    base_image="python:3.12",
)
def clean_up_dataset(
    session_id: str,
    models_mount_path: str = "/models",
) -> None:
    """Delete the saved calibration dataset from the PVC after compression."""

    import shutil
    from pathlib import Path

    dataset_path = Path(models_mount_path) / "sessions" / session_id / "dataset"

    if dataset_path.exists():
        print(f"Cleaning up dataset: {dataset_path}")
        shutil.rmtree(dataset_path)
        print("Dataset removed")
    else:
        print(f"Dataset directory does not exist (already cleaned up): {dataset_path}")
