from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git@main", "transformers", "datasets", "huggingface_hub"]
)
def compress_model(
    model_id: str,
    yaml_filename: str,
    session_id: str,
    model_name: str,
    num_calibration_samples: int,
    max_sequence_length: int,
    models_mount_path: str = "/models",
    yamls_mount_path: str = "/yamls",
) -> str:
    """
    Compress a model using LLM Compressor's oneshot API with a YAML recipe,
    run a smoke test generation, and save the compressed model to the PVC.
    Returns the model directory name used for saving.
    """

    import json
    import os
    import subprocess
    from pathlib import Path
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from compressed_tensors.offload import dispatch_model
    from huggingface_hub import snapshot_download
    from llmcompressor import oneshot
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

    # Load preprocessed calibration dataset (if available)
    dataset_path = Path(models_mount_path) / "sessions" / session_id / "dataset"
    ds = None
    if dataset_path.exists():
        print(f"Loading preprocessed dataset from {dataset_path}")
        ds = load_from_disk(str(dataset_path))
        print(f"Dataset loaded: {len(ds)} samples")
    else:
        import warnings
        warnings.warn(f"No preprocessed dataset found at {dataset_path}. Running oneshot without a dataset.")

    # Resolve recipe path
    recipe_path = str(Path(yamls_mount_path) / yaml_filename)
    print(f"Using recipe: {recipe_path}")

    # Triton needs a writable cache directory; the container's home may be /
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/.triton")

    # Run oneshot compression
    print(f"\n{'='*60}")
    print(f"Starting oneshot compression for {model_id}")
    print(f"{'='*60}")

    oneshot_kwargs = {
        "model": LOCAL_MODEL_PATH,
        "recipe": recipe_path,
    }
    if ds is not None:
        oneshot_kwargs["dataset"] = ds
        oneshot_kwargs["max_seq_length"] = max_sequence_length
        oneshot_kwargs["num_calibration_samples"] = num_calibration_samples

    compressed_model = oneshot(**oneshot_kwargs)

    print(f"\n{'='*60}")
    print("Compression complete, running smoke test")
    print(f"{'='*60}")

    # Smoke test generation
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    dispatch_model(compressed_model)

    input_ids = tokenizer(
        "Hello my name is", return_tensors="pt"
    ).input_ids.to(compressed_model.device)
    output = compressed_model.generate(input_ids, max_new_tokens=100)
    generated_text = tokenizer.decode(output[0])
    print(f"Smoke test output:\n{generated_text}")

    # Save compressed model
    save_path = Path(models_mount_path) / "sessions" / session_id / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving compressed model to {save_path}")
    compressed_model.save_pretrained(
        str(save_path), save_compressed=True, max_shard_size="5GB"
    )
    tokenizer.save_pretrained(str(save_path))

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
