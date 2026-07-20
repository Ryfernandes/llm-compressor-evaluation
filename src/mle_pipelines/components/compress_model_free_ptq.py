from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git@main", "huggingface_hub"]
)
def compress_model(
    model_id: str,
    scheme: str,
    session_id: str,
    model_name: str,
    ignore: str = "",
    max_workers: int = 15,
    models_mount_path: str = "/models",
) -> str:
    """
    Compress a model using LLM Compressor's model_free_ptq API with a
    quantization scheme preset, and save the compressed model to the PVC.
    Returns the session ID and model directory name.
    """

    import json
    import os
    import subprocess
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from llmcompressor import model_free_ptq
    from importlib.metadata import version as get_version

    llmcompressor_version = get_version("llmcompressor")

    ALLOW_PATTERNS = [
        "*.json",
        "*.safetensors",
        "*.model",
        "*.txt",
        "*.jinja",
        "tokenizer*",
        "special_tokens_map.json",
    ]

    IGNORE_PATTERNS = [
        "original/**/*",
        "*.bin",
        "*.gguf",
    ]

    LOCAL_MODEL_PATH = "/tmp/model"

    # Download model to a local directory (no HF cache)
    print(f"Downloading model {model_id} to {LOCAL_MODEL_PATH}...")
    snapshot_download(
        repo_id=model_id,
        token=os.environ.get("HF_TOKEN"),
        local_dir=LOCAL_MODEL_PATH,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"Model downloaded to {LOCAL_MODEL_PATH}")

    # Parse ignore list from comma-separated string
    ignore_list = [s.strip() for s in ignore.split(",") if s.strip()] if ignore else []
    if ignore_list:
        print(f"Ignore list: {ignore_list}")

    # Determine save path
    save_path = Path(models_mount_path) / "sessions" / session_id / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/.triton")

    # Run model-free PTQ compression
    print(f"\n{'='*60}")
    print(f"Starting model_free_ptq compression for {model_id}")
    print(f"Scheme: {scheme}")
    print(f"Max workers: {max_workers}")
    print(f"{'='*60}")

    model_free_ptq(
        model_stub=LOCAL_MODEL_PATH,
        save_directory=str(save_path),
        scheme=scheme,
        ignore=ignore_list,
        max_workers=max_workers,
    )

    print(f"\n{'='*60}")
    print("Compression complete")
    print(f"{'='*60}")

    # Save llmcompressor metadata
    freeze_output = subprocess.run(
        ["pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    source = next(
        line for line in freeze_output.splitlines()
        if line.lower().startswith("llmcompressor")
    )

    inspect_output = subprocess.run(
        ["pip", "inspect"], capture_output=True, text=True, check=True
    ).stdout
    installed_packages = json.loads(inspect_output)["installed"]
    llmcompressor_pkg = next(
        pkg for pkg in installed_packages
        if pkg["metadata"]["name"].lower() == "llmcompressor"
    )

    llmcompressor_info = {
        "source": source,
        "version": llmcompressor_version,
        "details": llmcompressor_pkg.get("direct_url"),
    }
    with open(save_path / "llmcompressor_source.json", "w") as f:
        json.dump(llmcompressor_info, f, indent=2)

    print(f"Model saved as: {model_name}")

    return f"Session ID: {session_id}\nModel Name: {model_name}"
