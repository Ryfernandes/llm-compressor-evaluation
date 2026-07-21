def get_hardware_constants() -> dict[str, int]:
    return {
        "H100_VRAM_GI": 76,
        "COMPRESSION_MEMORY_MULTIPLIER": 3,
        "COMPRESSION_MEMORY_BUFFER_GI": 10,
        "EVALUATION_KV_CACHE_VRAM_BUFFER_GI": 20,
        "EVALUATION_MODEL_STORAGE_BUFFER_GI": 10,
    }

def get_openshift_constants() -> dict[str, str]:
    return {
        "NAMESPACE": "machine-learning",
        "SERVICE_ACCOUNT_NAME": "ml-workload",
        "HF_SECRET_NAME": "ryan-test-hf-hub-secret",
        "H100_SELECTOR_KEY": "node-role.kubernetes.io/up-h100mcp",
        "TIER1_STORAGE_CLASS": "lvms-h100-tier1-storage",
    }

def get_evaluation_constants() -> dict[str, int]:
    return {
        "DEFAULT_BASE_SEED": 1234,
        "DEFAULT_TIMEOUT": 3600
    }

def get_evaluation_allow_ignore() -> list[list[str]]:
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
        "*.bin",
        "*.gguf",
    ]
    return [ALLOW_PATTERNS, IGNORE_PATTERNS]