def get_hardware_constants() -> dict[str, int]:
    return {
        "H100_VRAM_GI": 76,
        "COMPRESSION_MEMORY_MULTIPLIER": 3,
        "COMPRESSION_MEMORY_BUFFER_GI": 10,
        "EVALUATION_MODEL_STORAGE_BUFFER_GI": 10,
        "EVALUATION_KV_CACHE_VRAM_BUFFER_GI": 20
    }

def get_evaluation_constants() -> dict[str, int]:
    return {
        "DEFAULT_BASE_SEED": 1234,
        "DEFAULT_TIMEOUT": 3600
    }