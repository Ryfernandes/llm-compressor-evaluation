from kfp import dsl

@dsl.component(
    base_image="python:3.12",
)
def save_startup_statistics(
    # Session spec
    session_id: str,
    # Config spec
    config_filename: str,
    # PVC spec
    artifacts_pvc_mount_path: str,
    configs_pvc_mount_path: str,
    # Logs spec
    logs_filename: str = "vllm_server.log",
    evaluation_statistics_filename: str = "vllm_log_statistics.json",
) -> None:
    """Parse vLLM startup logs and save statistics to JSON."""
    import re
    import json
    from pathlib import Path

    session_path = Path(artifacts_pvc_mount_path) / "evaluation-artifacts" / "sessions" / session_id
    logs_path = session_path / "logs"

    log_file_path = logs_path / logs_filename
    stats_file_path = logs_path / evaluation_statistics_filename

    if not log_file_path.exists():
        raise FileNotFoundError(f"Log file not found at {log_file_path}")

    # Load config to get max_model_len
    config_path = Path(configs_pvc_mount_path) / config_filename
    with open(config_path, 'r') as f:
        config = json.load(f)
    max_model_len = config["model"]["max_model_len"]

    # Read the log file
    with open(log_file_path, 'r') as f:
        log_lines = f.readlines()

    stats = {}
    found_stats = set()

    # Define regex patterns for all statistics we need to extract
    # Each pattern maps to (regex, extractor_function, stats_keys_it_provides)
    patterns = {
        'version': (
            r'version ([\d.]+)',
            lambda m: {'vllm_version': m.group(1)},
            {'vllm_version'}
        ),
        'model_loading': (
            r'Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds',
            lambda m: {'model_size': float(m.group(1)), 'model_load_time': float(m.group(2))},
            {'model_size', 'model_load_time'}
        ),
        'kv_cache_memory': (
            r'Available KV cache memory: ([\d.]+) GiB',
            lambda m: {'kv_cache_size': float(m.group(1))},
            {'kv_cache_size'}
        ),
        'kv_cache_tokens': (
            r'GPU KV cache size: ([\d,]+) tokens',
            lambda m: {'kv_cache_tokens': int(m.group(1).replace(',', ''))},
            {'kv_cache_tokens'}
        ),
        'recommended_concurrency': (
            r'Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x',
            lambda m: {'recommended_concurrency': float(m.group(1))},
            {'recommended_concurrency'}
        ),
    }

    required_stats = {'vllm_version', 'model_size', 'kv_cache_size', 'kv_cache_tokens', 'recommended_concurrency', 'model_load_time'}

    # Single pass through log lines
    for line in log_lines:
        for pattern_name, (regex, extractor, stat_keys) in patterns.items():
            # Skip if we've already found all stats this pattern provides
            if stat_keys.issubset(found_stats):
                continue

            match = re.search(regex, line)
            if match:
                stats.update(extractor(match))
                found_stats.update(stat_keys)

        # Early exit if all required statistics are found
        if found_stats == required_stats:
            break

    # Validate that all required statistics were found
    missing_stats = [stat for stat in required_stats if stat not in stats]

    if missing_stats:
        raise ValueError(
            f"Failed to parse the following statistics from vLLM logs: {', '.join(missing_stats)}. "
            f"This likely indicates that vLLM has changed its logging format. "
            f"Please review the log file at {log_file_path} and update the parsing logic in save_startup_statistics()."
        )

    # Add max_model_len from config
    stats['max_model_len'] = max_model_len

    # Create the output structure
    output = {
        "last_read_line": len(log_lines),
        "start": stats
    }

    # Write to JSON file
    with open(stats_file_path, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"Startup statistics saved to {stats_file_path}")
    print(f"Parsed statistics: {stats}")
