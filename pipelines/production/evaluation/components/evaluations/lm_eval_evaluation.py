from kfp import dsl

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["lm-eval[api]", "requests", "prometheus-client"]
)
def lm_eval_evaluation(
    service_url: str,
    config_filename: str,
    session_id: str,
    model_path: str = "Qwen/Qwen3-8B",
    artifacts_pvc_mount_path: str = "/artifacts",
    configs_pvc_mount_path: str = "/configs",
) -> None:
    import os
    import subprocess
    import requests
    import json
    import time
    from pathlib import Path
    from prometheus_client.parser import text_string_to_metric_families

    # Default constants
    DEFAULT_BASE_SEED = 1234
    DEFAULT_TIMEOUT = 1800

    # Load configuration file
    config_path = Path(configs_pvc_mount_path) / config_filename
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract model configuration
    model_config = config.get("model", {})
    reasoning_parser = model_config.get("reasoning_parser", "")
    temperature = model_config.get("temperature", 0.6)
    top_p = model_config.get("top_p", 0.9)
    top_k = model_config.get("top_k", 50)
    max_model_len = model_config.get("max_model_len", 16384)

    # Extract and validate tasks for lm_eval harness
    all_tasks = config.get("tasks", [])
    task_list = []

    for task in all_tasks:
        if task.get("harness") != "lm_eval":
            continue

        # Check required fields
        required_fields = ["tag", "shots", "reps", "concurrency", "max_tokens"]
        missing_fields = [field for field in required_fields if field not in task]

        if missing_fields:
            print(f"WARNING: Skipping task due to missing required fields {missing_fields}: {task}")
            continue

        if task["max_tokens"] > max_model_len:
            print(f"WARNING: Skipping task {task['tag']} because max_tokens ({task['max_tokens']}) exceeds model's max_model_len ({max_model_len})")
            continue

        # Extract parameters with defaults for optional fields
        task_params = {
            "tag": task["tag"],
            "shots": task["shots"],
            "reps": task["reps"],
            "concurrency": task["concurrency"],
            "max_tokens": task["max_tokens"],
            "base_seed": task.get("base_seed", DEFAULT_BASE_SEED),
            "timeout": task.get("timeout", DEFAULT_TIMEOUT),
        }

        task_list.append(task_params)

    if not task_list:
        print("WARNING: No valid lm_eval tasks found in config. Exiting.")
        return

    print(f"Found {len(task_list)} lm_eval tasks to evaluate")
    for task in task_list:
        print(f"  - {task['tag']}: shots={task['shots']}, reps={task['reps']}, max_tokens={task['max_tokens']}")

    # Setup paths
    session_dir = Path(artifacts_pvc_mount_path) / "evaluation-artifacts" / "sessions" / session_id
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = session_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = session_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    # Setup vLLM metrics log file
    vllm_metrics_log = logs_dir / "vllm_metrics.jsonl"

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

    try:
        for task_params in task_list:
            task_tag = task_params["tag"]
            n_shots = task_params["shots"]
            max_gen_toks = task_params["max_tokens"]
            reps = task_params["reps"]
            num_concurrent = task_params["concurrency"]
            base_seed = task_params["base_seed"]
            timeout = task_params["timeout"]

            print(f"\n{'='*60}")
            print(f"Evaluating task: {task_tag}")
            print(f"{'='*60}\n")
            print(f"Configuration: shots={n_shots}, max_gen_toks={max_gen_toks}, max_length={max_model_len}, reps={reps}, concurrency={num_concurrent}, timeout={timeout}")

            for i in range(1, reps + 1):
                print(f"Evaluation Run {i}/{reps}")

                seed = base_seed + i

                # Configure proxy for this task/seed
                set_proxy_task(task_tag, seed)
                enable_proxy_logging()

                # Collect vLLM metrics before evaluation
                start_metrics = get_vllm_metrics()
                start_time_eval = time.time()

                # Build lm_eval command
                cmd = [
                    "python", "-m", "lm_eval",
                    "--model", "local-chat-completions",
                    "--tasks", task_tag,
                    "--model_args", (
                        f"model={model_path},"
                        f"max_length={max_model_len},"
                        f"base_url={base_url}/chat/completions,"
                        f"num_concurrent={num_concurrent},"
                        f"max_retries=3,"
                        f"tokenized_requests=False,"
                        f"tokenizer_backend=None,"
                        f"timeout={timeout}"
                    ),
                    "--num_fewshot", str(n_shots),
                    "--apply_chat_template",
                    "--fewshot_as_multiturn",
                    "--output_path", str(tmp_dir),
                    "--seed", str(seed),
                    "--gen_kwargs", (
                        f"do_sample=True,"
                        f"temperature={temperature},"
                        f"top_p={top_p},"
                        f"top_k={top_k},"
                        f"max_gen_toks={max_gen_toks},"
                        f"seed={seed}"
                    ),
                ]

                # Run evaluation
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)

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

                print(f"Evaluation complete, tried moving output to {str(tmp_dir)}")

                json_files = list(tmp_dir.rglob("*.json"))

                if json_files:
                    # Use the first (or only) results file found
                    json_file = json_files[0]
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