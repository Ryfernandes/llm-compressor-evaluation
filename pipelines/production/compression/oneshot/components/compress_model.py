from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["llmcompressor", "transformers", "datasets", "huggingface_hub"]
)
def compress_model(
    model_id: str,
    yaml_filename: str,
    session_id: str,
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

    import os
    import yaml
    from pathlib import Path
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from compressed_tensors.offload import dispatch_model
    from huggingface_hub import snapshot_download
    from llmcompressor import oneshot

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

    # Load preprocessed calibration dataset
    dataset_path = Path(models_mount_path) / "sessions" / session_id / "dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found: {dataset_path}")

    print(f"Loading preprocessed dataset from {dataset_path}")
    ds = load_from_disk(str(dataset_path))
    print(f"Dataset loaded: {len(ds)} samples")

    # Resolve recipe path
    recipe_path = str(Path(yamls_mount_path) / yaml_filename)
    print(f"Using recipe: {recipe_path}")

    # Run oneshot compression
    print(f"\n{'='*60}")
    print(f"Starting oneshot compression for {model_id}")
    print(f"{'='*60}")

    compressed_model = oneshot(
        model=LOCAL_MODEL_PATH,
        recipe=recipe_path,
        dataset=ds,
        max_seq_length=max_sequence_length,
        num_calibration_samples=num_calibration_samples,
    )

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

    # Determine model name from recipe scheme
    with open(recipe_path, "r") as f:
        recipe_data = yaml.safe_load(f)

    scheme = None
    for stage_content in recipe_data.values():
        if not isinstance(stage_content, dict):
            continue
        for group_content in stage_content.values():
            if not isinstance(group_content, dict):
                continue
            if "scheme" in group_content:
                scheme = group_content["scheme"]
                break
            for modifier_config in group_content.values():
                if isinstance(modifier_config, dict) and "scheme" in modifier_config:
                    scheme = modifier_config["scheme"]
                    break
            if scheme:
                break
        if scheme:
            break

    model_short_name = model_id.rstrip("/").split("/")[-1]
    if scheme:
        suffix = scheme.replace("_", "-")
    else:
        suffix = Path(yaml_filename).stem
    model_name = f"{model_short_name}-{suffix}"

    # Save compressed model
    save_path = Path(models_mount_path) / "sessions" / session_id / model_name
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving compressed model to {save_path}")
    compressed_model.save_pretrained(str(save_path), save_compressed=True)
    tokenizer.save_pretrained(str(save_path))
    print(f"Model saved as: {model_name}")

    return model_name
