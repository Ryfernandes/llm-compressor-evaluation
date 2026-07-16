from kfp import dsl

@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "lighteval[extended] @ git+https://github.com/neuralmagic/lighteval.git@eldar-fix-litellm",
        "litellm[caching]>=1.66.0",
        "requests",
        "prometheus-client",
        "pillow"
    ]
)
def lighteval_evaluation(
    service_url: str,
    model_config_name: str,
    evaluation_config_name: str,
    session_id: str,
    model_path: str = "Qwen/Qwen3-8B",
    artifacts_pvc_mount_path: str = "/artifacts",
    configs_pvc_mount_path: str = "/configs",
    packages_pvc_mount_path: str = "/packages",
) -> None:
    import os
    import subprocess
    import requests
    import json
    import time
    from pathlib import Path
    from prometheus_client.parser import text_string_to_metric_families

    # Default constants for optional fields
    DEFAULT_BASE_SEED = 1234
    DEFAULT_TIMEOUT = 3600

    # Load model configuration (already validated by validate_config component)
    model_config_path = Path(configs_pvc_mount_path) / "model" / model_config_name

    with open(model_config_path, 'r') as f:
        model_config = json.load(f)

    # Extract model configuration (all required fields guaranteed by validation)
    temperature = model_config["temperature"]
    top_p = model_config["top_p"]
    top_k = model_config["top_k"]
    max_model_len = model_config["max_model_len"]

    # Load evaluation configuration
    evaluation_config_path = Path(configs_pvc_mount_path) / "evaluation" / evaluation_config_name

    with open(evaluation_config_path, 'r') as f:
        evaluation_config = json.load(f)

    # Extract tasks for lighteval harness (required fields guaranteed by validation)
    all_tasks = evaluation_config["tasks"]
    task_list = []

    for task in all_tasks:
        if task["harness"] != "lighteval":
            continue

        # Extract parameters with defaults for optional fields
        # limit: if 0 or not present, set to None (no limit)
        limit = task.get("limit")
        if limit == 0:
            limit = None

        task_params = {
            "tag": task["tag"],
            "shots": task["shots"],
            "reps": task["reps"],
            "concurrency": task["concurrency"],
            "max_tokens": task["max_tokens"],
            "base_seed": task.get("base_seed", DEFAULT_BASE_SEED),
            "timeout": task.get("timeout", DEFAULT_TIMEOUT),
            "limit": limit,
        }

        task_list.append(task_params)

    if not task_list:
        print("INFO: No lighteval tasks found in config. Exiting.")
        return

    print(f"Found {len(task_list)} lighteval tasks to evaluate")
    for task in task_list:
        print(f"  - {task['tag']}: shots={task['shots']}, reps={task['reps']}, max_tokens={task['max_tokens']}")

    # Setup paths
    session_dir = Path(artifacts_pvc_mount_path) / "evaluation-artifacts" / "sessions" / session_id
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = session_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = session_dir / "tmp_lighteval"
    tmp_dir.mkdir(exist_ok=True)

    # Setup shared NLTK data directory for tasks requiring NLTK (ifeval, ifbench, math tasks)
    nltk_data_dir = Path(packages_pvc_mount_path) / "lighteval_evaluation" / "nltk_data"
    nltk_data_dir.mkdir(parents=True, exist_ok=True)

    # Set NLTK_DATA environment variable to use our writable directory
    os.environ["NLTK_DATA"] = str(nltk_data_dir)

    # Setup vLLM metrics log file
    vllm_metrics_log = logs_dir / "vllm_metrics.jsonl"

    # Setup vLLM log statistics file
    vllm_log_stats_file = logs_dir / "vllm_log_statistics.json"
    vllm_server_log_file = logs_dir / "vllm_server.log"

    # Setup harness metadata file
    harness_metadata_file = logs_dir / "harness_metadata.json"

    # Get lighteval version and install metadata from pip
    import importlib.metadata
    lighteval_version = importlib.metadata.version("lighteval")

    freeze_output = subprocess.run(
        ["pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    harness_name = next(
        line for line in freeze_output.splitlines()
        if line.lower().startswith("lighteval")
    )

    inspect_output = subprocess.run(
        ["pip", "inspect"], capture_output=True, text=True, check=True
    ).stdout
    installed_packages = json.loads(inspect_output)["installed"]
    lighteval_pkg = next(
        pkg for pkg in installed_packages
        if pkg["metadata"]["name"].lower() == "lighteval"
    )
    harness_details = lighteval_pkg.get("direct_url")

    print(f"Using {harness_name} version {lighteval_version}")

    # Change to session directory
    os.chdir(session_dir)

    # Ensure service_url uses the correct format for base_url
    base_url = service_url.rstrip("/") + "/v1"

    # Helper functions for proxy control
    def set_proxy_task(task_id: str, seed: int):
        """Set the task_id and seed in the proxy."""
        response = requests.post(
            f"{service_url}/set-proxy-task",
            json={"task_id": task_id, "seed": seed},
            timeout=10
        )
        response.raise_for_status()
        print(f"Proxy task set: task_id={task_id}, seed={seed}")

    def enable_proxy_logging():
        """Enable logging in the proxy."""
        response = requests.post(f"{service_url}/start-logging", timeout=10)
        response.raise_for_status()
        print("Proxy logging enabled")

    def disable_proxy_logging():
        """Disable logging in the proxy."""
        response = requests.post(f"{service_url}/stop-logging", timeout=10)
        response.raise_for_status()
        print("Proxy logging disabled")

    def get_vllm_metrics():
        """Fetch and parse vLLM Prometheus metrics."""
        try:
            response = requests.get(f"{service_url}/metrics", timeout=30)
            response.raise_for_status()

            metrics = {}
            available_metrics = []

            for family in text_string_to_metric_families(response.text):
                available_metrics.append(family.name)

                # Queue time histogram metrics (uses colon namespace)
                if family.name == "vllm:request_queue_time_seconds":
                    for sample in family.samples:
                        if sample.name == "vllm:request_queue_time_seconds_sum":
                            metrics["queue_time_sum"] = sample.value
                        elif sample.name == "vllm:request_queue_time_seconds_count":
                            metrics["queue_time_count"] = sample.value

                # Counter metrics (note: no _total suffix in family name, but may be in sample name)
                elif family.name == "vllm:num_preemptions":
                    for sample in family.samples:
                        # Sample name might be vllm:num_preemptions_total or vllm:num_preemptions
                        metrics["preemptions_total"] = sample.value
                        break  # Take first sample

                elif family.name == "vllm:prompt_tokens":
                    for sample in family.samples:
                        # Sample name might be vllm:prompt_tokens_total or vllm:prompt_tokens
                        metrics["prompt_tokens_total"] = sample.value
                        break  # Take first sample

                elif family.name == "vllm:generation_tokens":
                    for sample in family.samples:
                        # Sample name might be vllm:generation_tokens_total or vllm:generation_tokens
                        metrics["generation_tokens_total"] = sample.value
                        break  # Take first sample

            # Debug: Print available metrics if any expected ones are missing
            required_metrics = ["queue_time_sum", "queue_time_count", "preemptions_total",
                              "prompt_tokens_total", "generation_tokens_total"]
            missing = [m for m in required_metrics if m not in metrics]

            if missing:
                raise ValueError(f"Critical vLLM metrics missing: {missing}. Available: {sorted(set(available_metrics))}")

            return metrics
        except Exception as e:
            print(f"ERROR: Failed to fetch vLLM metrics: {e}")
            raise

    def compute_metric_deltas(start_metrics, end_metrics, duration_seconds):
        """Compute deltas between start and end metrics."""
        if not start_metrics or not end_metrics:
            return None

        delta = {}

        # Queue time metrics
        queue_time_sum_delta = end_metrics.get("queue_time_sum", -1) - start_metrics.get("queue_time_sum", 0)
        queue_time_count_delta = end_metrics.get("queue_time_count", -1) - start_metrics.get("queue_time_count", 0)

        delta["queue_time_sum_seconds"] = queue_time_sum_delta
        delta["queue_time_count"] = queue_time_count_delta
        delta["queue_time_avg_seconds"] = queue_time_sum_delta / queue_time_count_delta if queue_time_count_delta > 0 else 0

        # Preemptions
        delta["preemptions_total"] = end_metrics.get("preemptions_total", -1) - start_metrics.get("preemptions_total", 0)

        # Token counts
        prompt_tokens_delta = end_metrics.get("prompt_tokens_total", -1) - start_metrics.get("prompt_tokens_total", 0)
        generation_tokens_delta = end_metrics.get("generation_tokens_total", -1) - start_metrics.get("generation_tokens_total", 0)
        total_tokens_delta = prompt_tokens_delta + generation_tokens_delta

        delta["prompt_tokens_total"] = prompt_tokens_delta
        delta["generation_tokens_total"] = generation_tokens_delta
        delta["total_tokens"] = total_tokens_delta
        delta["duration_seconds"] = duration_seconds

        return delta

    def parse_kv_cache_utilization(task_id: str, seed: int):
        """Parse KV cache utilization from vLLM server logs and update statistics file."""
        import re

        # Load existing statistics file or create new structure
        if vllm_log_stats_file.exists():
            with open(vllm_log_stats_file, 'r') as f:
                stats_data = json.load(f)
        else:
            # Initialize with empty structure if file doesn't exist yet
            stats_data = {
                "last_read_line": 0,
                "start": {},
                "tasks": {}
            }

        # Read vLLM server log from last read position
        if not vllm_server_log_file.exists():
            print(f"WARNING: vLLM server log not found at {vllm_server_log_file}")
            return

        with open(vllm_server_log_file, 'r') as f:
            all_lines = f.readlines()

        last_read_line = stats_data.get("last_read_line", 0)
        new_lines = all_lines[last_read_line:]

        # Parse engine log lines for KV cache and throughput metrics
        # Example: "INFO 07-01 20:44:04 [loggers.py:271] Engine 000: Avg prompt throughput: 2334.2 tokens/s, Avg generation throughput: 12900.7 tokens/s, Running: 511 reqs, Waiting: 0 reqs, GPU KV cache usage: 54.6%, Prefix cache hit rate: 87.5%"
        engine_log_pattern = re.compile(
            r'INFO\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+.*?Engine \d+:\s+'
            r'Avg prompt throughput:\s+([\d.]+)\s+tokens/s,\s+'
            r'Avg generation throughput:\s+([\d.]+)\s+tokens/s,\s+'
            r'Running:\s+(\d+)\s+reqs,\s+'
            r'Waiting:\s+(\d+)\s+reqs,\s+'
            r'GPU KV cache usage:\s+([\d.]+)%,\s+'
            r'Prefix cache hit rate:\s+([\d.]+)%'
        )

        samples = []
        for line in new_lines:
            match = engine_log_pattern.search(line)
            if match:
                sample = {
                    "timestamp": match.group(1),
                    "prompt_throughput": float(match.group(2)),
                    "generation_throughput": float(match.group(3)),
                    "running_reqs": int(match.group(4)),
                    "waiting_reqs": int(match.group(5)),
                    "kv_cache_usage_pct": float(match.group(6)),
                    "prefix_cache_hit_rate": float(match.group(7))
                }
                samples.append(sample)

        # Update statistics structure: tasks -> task_id -> seeds -> [samples]
        if "tasks" not in stats_data:
            stats_data["tasks"] = {}

        if task_id not in stats_data["tasks"]:
            stats_data["tasks"][task_id] = {"seeds": {}}

        seed_key = str(seed)
        if seed_key not in stats_data["tasks"][task_id]["seeds"]:
            stats_data["tasks"][task_id]["seeds"][seed_key] = []

        # Append new samples to this seed's data
        stats_data["tasks"][task_id]["seeds"][seed_key].extend(samples)

        # Update last read line
        stats_data["last_read_line"] = len(all_lines)

        # Write updated statistics
        with open(vllm_log_stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)

        print(f"Parsed {len(samples)} KV cache utilization samples for task={task_id}, seed={seed}")
        if samples:
            print(f"  KV cache usage range: {min(s['kv_cache_usage_pct'] for s in samples):.1f}% - {max(s['kv_cache_usage_pct'] for s in samples):.1f}%")

    try:
        for task_params in task_list:
            task_tag = task_params["tag"]
            n_shots = task_params["shots"]
            max_gen_toks = task_params["max_tokens"]
            reps = task_params["reps"]
            num_concurrent = task_params["concurrency"]
            base_seed = task_params["base_seed"]
            timeout = task_params["timeout"]
            limit = task_params["limit"]

            print(f"\n{'='*60}")
            print(f"Evaluating task: {task_tag}")
            print(f"{'='*60}\n")
            print(f"Configuration: shots={n_shots}, max_gen_toks={max_gen_toks}, max_length={max_model_len}, reps={reps}, concurrency={num_concurrent}, timeout={timeout}, limit={limit}")

            for i in range(1, reps + 1):
                print(f"Evaluation Run {i}/{reps}")

                seed = base_seed + i

                # Configure proxy for this task/seed
                # Note: LightEval task IDs include |0 suffix, but we use the base tag for tracking
                set_proxy_task(task_tag, seed)
                enable_proxy_logging()

                # Collect vLLM metrics before evaluation
                start_metrics = get_vllm_metrics()
                start_time_eval = time.time()

                # Build lighteval command
                # Format: lighteval endpoint litellm "model_name=...,provider=...,base_url=...,..." "task@k=1@n=1|0" --output-dir results/... --save-details
                model_args = (
                    f"model_name=hosted_vllm/{model_path},"
                    f"provider=hosted_vllm,"
                    f"base_url={base_url},"
                    f"timeout={timeout},"
                    f"concurrent_requests={num_concurrent},"
                    f"generation_parameters={{temperature:{temperature},max_new_tokens:{max_gen_toks},top_p:{top_p},seed:{seed},top_k:{top_k}}}"
                )

                # LightEval task format: task_tag@k=1@n=1|0
                # The |0 suffix indicates the first (and usually only) variant
                task_spec = f"{task_tag}@k=1@n=1|0"

                cmd = [
                    "python", "-m", "lighteval", "endpoint", "litellm",
                    model_args,
                    task_spec,
                    "--remove-reasoning-tags",
                    "--output-dir", str(tmp_dir),
                    "--save-details",
                ]

                # Add --max-samples argument if limit is set
                if limit is not None:
                    cmd.extend(["--max-samples", str(limit)])

                # Run evaluation
                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(result.stdout)
                    if result.stderr:
                        print("STDERR:", result.stderr)
                except subprocess.CalledProcessError as e:
                    print(f"ERROR: lighteval command failed with exit code {e.returncode}")
                    print("STDOUT:", e.stdout)
                    print("STDERR:", e.stderr)
                    raise

                # Collect vLLM metrics after evaluation
                end_time_eval = time.time()
                end_metrics = get_vllm_metrics()
                duration_seconds = end_time_eval - start_time_eval

                # Compute metric deltas
                metric_deltas = compute_metric_deltas(start_metrics, end_metrics, duration_seconds)

                # Log metrics to JSONL
                if metric_deltas:
                    log_entry = {
                        "task_id": task_tag,
                        "seed": seed,
                        "metrics": metric_deltas,
                        "timestamp": end_time_eval,
                    }
                    with open(vllm_metrics_log, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
                    print(f"vLLM metrics logged: {metric_deltas}")

                # Disable proxy logging after evaluation completes
                disable_proxy_logging()

                # Parse KV cache utilization from vLLM server logs
                parse_kv_cache_utilization(task_tag, seed)

                # Track harness metadata for this task
                if harness_metadata_file.exists():
                    with open(harness_metadata_file, 'r') as f:
                        harness_metadata = json.load(f)
                else:
                    harness_metadata = {}

                if task_tag not in harness_metadata:
                    harness_metadata[task_tag] = {
                        "harness": harness_name,
                        "version": lighteval_version,
                        "harness_details": harness_details,
                    }

                with open(harness_metadata_file, 'w') as f:
                    json.dump(harness_metadata, f, indent=2)

                print(f"Evaluation complete, tried moving output to {str(tmp_dir)}")

                # Search for any JSON files in tmp directory
                json_files = list(tmp_dir.rglob("*.json"))

                if json_files:
                    # Use the first (or only) results file found
                    json_file = json_files[0]

                    # Extract timestamp from lighteval's output filename (e.g., results_2026-07-06T08-03-46.941709.json)
                    # and use it in our output filename to preserve it for collation
                    original_name = json_file.name
                    import re
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)', original_name)

                    if timestamp_match:
                        timestamp = timestamp_match.group(1)
                        output_name = f"{task_tag}_seed_{seed}_{timestamp}.json"
                    else:
                        # Fallback if no timestamp found (shouldn't happen with lighteval)
                        output_name = f"{task_tag}_seed_{seed}.json"

                    output_path = results_dir / output_name

                    # Copy instead of rename to avoid cross-device link errors
                    import shutil
                    shutil.copy2(json_file, output_path)
                    print(f"Saved results to: {output_path}")
                    print(f"Source file: {json_file}")
                else:
                    print(f"WARNING: No JSON output found for {task_tag} run {i}")
                    # Debug: list what's actually in tmp_dir
                    print(f"Contents of {tmp_dir}:")
                    for item in tmp_dir.rglob("*"):
                        print(f"  {item}")

                # Clean tmp directory
                for file in tmp_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        import shutil
                        shutil.rmtree(file)

        print(f"\n{'='*60}")
        print(f"All evaluations complete. Results saved to: {results_dir}")
        print(f"{'='*60}")

    finally:
        # Cleanup tmp directory
        if tmp_dir.exists():
            import shutil
            shutil.rmtree(tmp_dir)
