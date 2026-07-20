from mle_pipelines.utils.decorators import hardware_component


@hardware_component(
    base_image="python:3.12",
    packages_to_install=["huggingface_hub"]
)
def get_model_size(
    model_id: str
) -> str:
    """
    Determine model size via a dry-run HuggingFace download, validate
    that it fits on a single H100 GPU, and return the recommended
    memory request string for the compression step.

    The memory request is calculated as:
        ceil(model_size_gi * memory_multiplier) + memory_buffer_gi
    """

    import math
    import os
    from huggingface_hub import snapshot_download

    constants = get_hardware_constants()

    H100_VRAM_GI = constants["H100_VRAM_GI"]
    COMPRESSION_MEMORY_MULTIPLIER = constants["COMPRESSION_MEMORY_MULTIPLIER"]
    COMPRESSION_MEMORY_BUFFER_GI = constants["COMPRESSION_MEMORY_BUFFER_GI"]

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

    print(f"Running dry-run download to estimate size of {model_id}...")

    dry_run_files = snapshot_download(
        repo_id=model_id,
        token=os.environ.get("HF_TOKEN"),
        dry_run=True,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )

    model_size_bytes = sum(
        f.file_size for f in dry_run_files
        if f.file_size is not None
    )
    model_size_gi = model_size_bytes / (1024 ** 3)

    print(f"Model size: {model_size_gi:.2f} GiB")

    if model_size_gi > H100_VRAM_GI:
        raise ValueError(
            f"Model size {model_size_gi:.1f} GiB exceeds single H100 GPU "
            f"capacity ({H100_VRAM_GI} GiB). Multi-GPU compression is not "
            f"supported by this pipeline."
        )

    compress_memory_gi = int(math.ceil(model_size_gi * COMPRESSION_MEMORY_MULTIPLIER)) + COMPRESSION_MEMORY_BUFFER_GI
    memory_request = f"{compress_memory_gi}Gi"

    print(f"Memory multiplier: {COMPRESSION_MEMORY_MULTIPLIER}x")
    print(f"Memory buffer: {COMPRESSION_MEMORY_BUFFER_GI} GiB")
    print(f"Compression memory request: {memory_request}")

    return memory_request
