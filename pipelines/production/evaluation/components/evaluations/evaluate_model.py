from kfp import dsl

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["lm-eval[api]", "requests", "prometheus-client"]
)
def evaluate_model(
    service_url: str,
    tasks: str,
    session_id: str,
    reasoning_parser: str,
    model_path: str = "Qwen/Qwen3-8B",
    artifacts_pvc_mount_path: str = "/artifacts",
    num_concurrent: int = 128,
) -> None:
    import os
    import subprocess
    import requests
    import json
    import time
    from pathlib import Path
    from prometheus_client.parser import text_string_to_metric_families

    # Configuration registry based on task name and reasoning mode
    # Format: (task_name, reasoning_enabled): {n_shots, max_gen_toks, max_length, reps}
    # max_length = max_gen_toks + 8192 (context buffer)
    EVAL_CONFIG = {
        # Non-reasoning configurations
        ("gsm8k_platinum_cot_llama", False): {"n_shots": 5, "max_gen_toks": 8192, "max_length": 16384, "reps": 3},
        ("mmlu_cot_llama", False): {"n_shots": 5, "max_gen_toks": 8192, "max_length": 16384, "reps": 3},
        ("mmlu_pro_chat", False): {"n_shots": 5, "max_gen_toks": 8192, "max_length": 16384, "reps": 3},
        ("ifeval", False): {"n_shots": 0, "max_gen_toks": 8192, "max_length": 16384, "reps": 3},
        ("math_500", False): {"n_shots": 0, "max_gen_toks": 8192, "max_length": 16384, "reps": 3},
        ("lcb:codegeneration_v6", False): {"n_shots": 0, "max_gen_toks": 8192, "max_length": 16384, "reps": 3},
        ("mrcr", False): {"n_shots": 0, "max_gen_toks": 8192, "max_length": 16384, "reps": 3},

        # Reasoning-enabled configurations
        ("gsm8k_platinum_cot_llama", True): {"n_shots": 0, "max_gen_toks": 32000, "max_length": 40192, "reps": 3},
        ("mmlu_pro_chat", True): {"n_shots": 0, "max_gen_toks": 32000, "max_length": 40192, "reps": 3},
        ("ifeval", True): {"n_shots": 0, "max_gen_toks": 32000, "max_length": 40192, "reps": 3},
        ("math_500", True): {"n_shots": 0, "max_gen_toks": 32000, "max_length": 40192, "reps": 3},
        ("aime25", True): {"n_shots": 0, "max_gen_toks": 32000, "max_length": 40192, "reps": 8},
        ("gpqa:diamond", True): {"n_shots": 0, "max_gen_toks": 32000, "max_length": 40192, "reps": 3},
        ("lcb:codegeneration_v6", True): {"n_shots": 0, "max_gen_toks": 32000, "max_length": 40192, "reps": 3},
        ("mrcr", True): {"n_shots": 0, "max_gen_toks": 32000, "max_length": 40192, "reps": 3},
    }

    # Default configuration if task not found
    DEFAULT_CONFIG = {"n_shots": 5, "max_gen_toks": 4096, "max_length": 32768, "reps": 1}

    # Determine if reasoning is enabled
    reasoning_enabled = bool(reasoning_parser and reasoning_parser.strip())

    # Parse and clean task list
    task_list = [task.strip() for task in tasks.split(",") if task.strip()]

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

        # Throughput (tokens per second)
        delta["throughput_tokens_per_second"] = total_tokens_delta / duration_seconds if duration_seconds > 0 else 0
        delta["duration_seconds"] = duration_seconds

        return delta

    try:
        for task_tag in task_list:
            print(f"\n{'='*60}")
            print(f"Evaluating task: {task_tag}")
            print(f"{'='*60}\n")

            # Get configuration for this task
            config_key = (task_tag, reasoning_enabled)
            config = EVAL_CONFIG.get(config_key, DEFAULT_CONFIG)

            n_shots = config["n_shots"]
            max_gen_toks = config["max_gen_toks"]
            max_length = config["max_length"]
            reps = config["reps"]

            print(f"Configuration: n_shots={n_shots}, max_gen_toks={max_gen_toks}, max_length={max_length}, reps={reps}, reasoning_enabled={reasoning_enabled}")

            for i in range(1, reps + 1):
                print(f"Evaluation Run {i}/{reps}")

                seed = 1233 + i

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
                        f"max_length={max_length},"
                        f"base_url={base_url}/chat/completions,"
                        f"num_concurrent={num_concurrent},"
                        f"max_retries=3,"
                        f"tokenized_requests=False,"
                        f"tokenizer_backend=None,"
                        f"timeout=1200"
                    ),
                    "--num_fewshot", str(n_shots),
                    "--apply_chat_template",
                    "--fewshot_as_multiturn",
                    "--output_path", str(tmp_dir),
                    "--seed", str(seed),
                    "--gen_kwargs", (
                        f"do_sample=True,"
                        f"temperature=0.6,"
                        f"top_p=0.9,"
                        f"top_k=50,"
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