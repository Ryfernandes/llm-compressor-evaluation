from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["huggingface_hub"]
)
def upload_to_huggingface(
    session_id: str,
    model_name: str,
    models_mount_path: str = "/models",
) -> str:
    """
    Upload a compressed model to a new HuggingFace repository.
    """

    import os
    from pathlib import Path
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    model_path = Path(models_mount_path) / "sessions" / session_id / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Compressed model not found: {model_path}")

    api = HfApi(token=hf_token)
    user_info = api.whoami()
    username = user_info["name"]

    repo_id = f"{username}/{model_name}"
    print(f"Creating HuggingFace repository: {repo_id}")

    try:
        api.create_repo(repo_id=repo_id, repo_type="model")
    except HfHubHTTPError:
        repo_id = f"{username}/{model_name}-{session_id}"
        print(f"Repository name taken, using: {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="model")

    print(f"Uploading model from {model_path} to {repo_id}")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
    )

    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"Model uploaded successfully: {repo_url}")
    return repo_url
