from kfp import dsl


@dsl.component(base_image="python:3.12")
def collate_results(
    session_id: str,
    model_id: str,
    artifacts_pvc_mount_path: str = "/artifacts",
):
    """Collate evaluation results from model evaluation runs."""
    import json
    import os
    import statistics
    from collections import defaultdict
    from datetime import datetime, timezone
    from pathlib import Path

    # Metrics registry
    METRICS_REGISTRY = {
        "gsm8k": ["exact_match,strict-match"],
        "gsm8k_platinum_cot_llama": ["exact_match,strict-match"],
        "mmlu_cot_llama": ["exact_match,strict_match"],
        "mmlu_pro_chat": ["exact_match,custom-extract"],
        "ifeval": ["inst_level_strict_acc,none"],
        "math_500|0": ["pass@k:k=1&n=1"],
        "aime25|0": ["pass@k:k=1&n=1"],
        "gpqa:diamond|0": ["gpqa_pass@k:k=1&n=1"],
        "lcb:codegeneration_v6": ["codegen_pass@k:k=1&n=1"],
        "mrcr": ["score_gt_16k_le_32k", "AUC"],
    }

    def scan_json_files(directories):
        """Recursively find all JSON files in directories with source tracking."""
        json_files = []
        for directory in directories:
            dir_path = Path(directory).resolve()
            if dir_path.exists() and dir_path.is_dir():
                # Track which source directory each file came from
                for f in dir_path.rglob("*.json"):
                    if f.is_file():
                        json_files.append((str(f), str(directory)))
        return sorted(json_files, key=lambda x: x[0])

    def detect_framework(json_data):
        """Detect if results are from LM-Eval or LightEval."""
        if "config_general" in json_data:
            return "lighteval"
        if all(key in json_data for key in ["results", "configs", "config"]):
            return "lmeval"
        return None

    def extract_lmeval_metrics(task_name, task_results):
        """Extract metrics from LM-Eval format."""
        metrics = {}
        if task_name in METRICS_REGISTRY:
            for metric_key in METRICS_REGISTRY[task_name]:
                value = task_results.get(metric_key)
                stderr_key = f"{metric_key}_stderr"
                stderr = task_results.get(stderr_key)
                if stderr == "N/A":
                    stderr = None
                if value is not None:
                    metrics[metric_key] = {"value": value, "stderr": stderr}
        return metrics

    def extract_lmeval_result(data, task_name, source_filename):
        """Extract single task result from LM-Eval."""
        task_results = data["results"][task_name]
        metrics = extract_lmeval_metrics(task_name, task_results)
        gen_kwargs = data.get("config", {}).get("gen_kwargs", {})

        timestamp = data.get("date")
        duration = data.get("total_evaluation_time_seconds")
        if isinstance(duration, str):
            try:
                duration = float(duration)
            except (ValueError, TypeError):
                duration = None

        return {
            "task_name": task_name,
            "model_name": data.get("model_name"),
            "source_filename": source_filename,
            "evaluation_datetime": timestamp,
            "evaluation_datetime_iso": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat() if timestamp else None,
            "evaluation_duration_seconds": duration,
            "metrics": metrics,
            "inference_parameters": {
                "do_sample": gen_kwargs.get("do_sample"),
                "temperature": gen_kwargs.get("temperature"),
                "top_p": gen_kwargs.get("top_p"),
                "top_k": gen_kwargs.get("top_k"),
                "max_gen_toks": gen_kwargs.get("max_gen_toks"),
                "seed": gen_kwargs.get("seed"),
            },
        }

    def parse_json_files(json_files):
        """Parse all JSON files and extract results."""
        all_results = []
        for json_file_path, source_dir in json_files:
            try:
                with open(json_file_path, "r") as f:
                    json_data = json.load(f)

                framework = detect_framework(json_data)
                if framework != "lmeval":
                    print(f"Skipping non-LM-Eval file: {json_file_path}")
                    continue

                source_filename = os.path.basename(json_file_path)

                for task_name in json_data.get("results", {}).keys():
                    if task_name in METRICS_REGISTRY:
                        result = extract_lmeval_result(json_data, task_name, source_filename)
                        result["source_directory"] = source_dir
                        all_results.append(result)
                    else:
                        print(f"Skipping unknown task '{task_name}' in {json_file_path}")
            except Exception as e:
                print(f"ERROR parsing {json_file_path}: {e}")
        return all_results

    def compute_aggregate_stats(runs):
        """Compute aggregate statistics across runs."""
        stats = {}

        durations = [r["evaluation_duration_seconds"] for r in runs if r["evaluation_duration_seconds"] is not None]
        stats["evaluation_duration_seconds"] = {
            "mean": statistics.mean(durations) if durations else None,
            "std": statistics.stdev(durations) if len(durations) > 1 else 0.0
        }

        all_metric_names = set()
        for run in runs:
            all_metric_names.update(run["metrics"].keys())

        metrics_stats = {}
        for metric_name in all_metric_names:
            values = [run["metrics"][metric_name]["value"] for run in runs if metric_name in run["metrics"] and run["metrics"][metric_name]["value"] is not None]
            stderrs = [run["metrics"][metric_name]["stderr"] for run in runs if metric_name in run["metrics"] and run["metrics"][metric_name]["stderr"] is not None]

            metrics_stats[metric_name] = {
                "value": {
                    "mean": statistics.mean(values) if values else None,
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0
                },
                "stderr": {
                    "mean": statistics.mean(stderrs) if stderrs else None,
                    "std": statistics.stdev(stderrs) if len(stderrs) > 1 else 0.0
                }
            }
        stats["metrics"] = metrics_stats
        return stats

    def parse_vllm_metrics(logs_path):
        """Parse vLLM metrics JSONL logs."""
        if not logs_path.exists():
            print(f"No vLLM metrics found at {logs_path}")
            return {}

        task_seed_metrics = {}
        try:
            with open(logs_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        task_id = entry.get("task_id")
                        seed = entry.get("seed")
                        metrics = entry.get("metrics")
                        if task_id is not None and seed is not None and metrics is not None:
                            task_seed_metrics[(task_id, seed)] = metrics
                    except json.JSONDecodeError as e:
                        print(f"ERROR parsing vLLM metrics line: {e}")
        except Exception as e:
            print(f"ERROR reading vLLM metrics: {e}")
            return {}

        return task_seed_metrics

    def aggregate_vllm_metrics_by_task(task_seed_metrics):
        """Aggregate vLLM metrics at the task level across all seeds."""
        # Group by task_id
        metrics_by_task = defaultdict(list)
        for (task_id, seed), metrics in task_seed_metrics.items():
            metrics_by_task[task_id].append(metrics)

        # Aggregate per task
        task_aggregates = {}
        for task_id, metrics_list in metrics_by_task.items():
            # Collect all values across seeds for numeric fields
            queue_time_sums = [m["queue_time_sum_seconds"] for m in metrics_list if "queue_time_sum_seconds" in m]
            queue_time_counts = [m["queue_time_count"] for m in metrics_list if "queue_time_count" in m]
            queue_time_avgs = [m["queue_time_avg_seconds"] for m in metrics_list if "queue_time_avg_seconds" in m]
            preemptions = [m["preemptions_total"] for m in metrics_list if "preemptions_total" in m]
            prompt_tokens = [m["prompt_tokens_total"] for m in metrics_list if "prompt_tokens_total" in m]
            generation_tokens = [m["generation_tokens_total"] for m in metrics_list if "generation_tokens_total" in m]
            total_tokens = [m["total_tokens"] for m in metrics_list if "total_tokens" in m]
            durations = [m["duration_seconds"] for m in metrics_list if "duration_seconds" in m]

            aggregate = {
                "queue_time_sum_seconds": sum(queue_time_sums) if queue_time_sums else 0,
                "queue_time_count": sum(queue_time_counts) if queue_time_counts else 0,
                "queue_time_avg_seconds": statistics.mean(queue_time_avgs) if queue_time_avgs else 0,
                "preemptions_total": sum(preemptions) if preemptions else 0,
                "prompt_tokens_total": sum(prompt_tokens) if prompt_tokens else 0,
                "generation_tokens_total": sum(generation_tokens) if generation_tokens else 0,
                "total_tokens": sum(total_tokens) if total_tokens else 0,
                "duration_seconds": sum(durations) if durations else 0,
                "num_seeds": len(metrics_list),
            }

            task_aggregates[task_id] = aggregate

        return task_aggregates

    def parse_proxy_logs(logs_path):
        """Parse proxy JSONL logs and compute statistics per task/seed."""
        if not logs_path.exists():
            print(f"No proxy logs found at {logs_path}")
            return {}, {}

        # Group logs by (task_id, seed) and by task_id alone
        logs_by_task_seed = defaultdict(list)
        logs_by_task = defaultdict(list)

        try:
            with open(logs_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        task_id = entry.get("task_id")
                        seed = entry.get("seed")
                        if task_id is not None and seed is not None:
                            logs_by_task_seed[(task_id, seed)].append(entry)
                            logs_by_task[task_id].append(entry)
                    except json.JSONDecodeError as e:
                        print(f"ERROR parsing proxy log line: {e}")
        except Exception as e:
            print(f"ERROR reading proxy logs: {e}")
            return {}, {}

        # Compute statistics per task/seed
        task_seed_stats = {}
        for (task_id, seed), entries in logs_by_task_seed.items():
            stats = compute_proxy_stats(entries)
            task_seed_stats[(task_id, seed)] = stats

        return task_seed_stats, logs_by_task

    def compute_proxy_stats(entries):
        """Compute min/max/mean/p95/p99 for tokens and latency, and finish_reason counts."""
        if not entries:
            return None

        # Extract numeric fields (filtering out None values)
        prompt_tokens = [e["prompt_tokens"] for e in entries if e.get("prompt_tokens") is not None]
        completion_tokens = [e["completion_tokens"] for e in entries if e.get("completion_tokens") is not None]
        total_tokens = [e["total_tokens"] for e in entries if e.get("total_tokens") is not None]
        latencies = [e["latency_seconds"] for e in entries if e.get("latency_seconds") is not None]

        def compute_percentile(values, percentile):
            """Compute percentile using sorted values."""
            if not values:
                return None
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            k = (n - 1) * percentile / 100.0
            f = int(k)
            c = k - f
            if f + 1 < n:
                return sorted_vals[f] * (1 - c) + sorted_vals[f + 1] * c
            return sorted_vals[f]

        def stats_dict(values):
            """Compute min/max/mean/p95/p99 for a list of values."""
            if not values:
                return None
            return {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "p95": compute_percentile(values, 95),
                "p99": compute_percentile(values, 99),
            }

        # Count finish reasons
        finish_reason_counts = defaultdict(int)
        for entry in entries:
            reason = entry.get("finish_reason", "unknown")
            finish_reason_counts[reason] += 1

        return {
            "prompt_tokens": stats_dict(prompt_tokens),
            "completion_tokens": stats_dict(completion_tokens),
            "total_tokens": stats_dict(total_tokens),
            "latency_seconds": stats_dict(latencies),
            "finish_reason_counts": dict(finish_reason_counts),
            "total_requests": len(entries),
        }

    def aggregate_proxy_stats_by_task(logs_by_task, task_seed_stats):
        """Aggregate proxy stats at the task level across all seeds."""
        task_aggregates = {}

        for task_id, entries in logs_by_task.items():
            # Compute statistics from all raw entries for this task
            stats = compute_proxy_stats(entries)

            # Count unique seeds for this task
            unique_seeds = set()
            for (tid, seed), _ in task_seed_stats.items():
                if tid == task_id:
                    unique_seeds.add(seed)

            # Add num_seeds field
            if stats:
                stats["num_seeds"] = len(unique_seeds)

            task_aggregates[task_id] = stats

        return task_aggregates

    def group_results(all_results, task_seed_proxy_stats, task_seed_vllm_metrics):
        """Group results by (task_name, model_name)."""
        groups = defaultdict(list)
        for result in all_results:
            key = (result["task_name"], result["model_name"])
            groups[key].append(result)

        grouped_results = []
        for (task_name, model_name), runs in groups.items():
            first_run = runs[0]
            run_data = []
            for r in runs:
                # Extract seed from inference_parameters
                seed = r["inference_parameters"].get("seed")

                # Find matching proxy stats for this task/seed
                proxy_stats = None
                if seed is not None and (task_name, seed) in task_seed_proxy_stats:
                    proxy_stats = task_seed_proxy_stats[(task_name, seed)]

                # Find matching vLLM metrics for this task/seed
                vllm_metrics = None
                if seed is not None and (task_name, seed) in task_seed_vllm_metrics:
                    vllm_metrics = task_seed_vllm_metrics[(task_name, seed)]

                run_data.append({
                    "source_filename": r["source_filename"],
                    "evaluation_datetime": r["evaluation_datetime"],
                    "evaluation_datetime_iso": r["evaluation_datetime_iso"],
                    "evaluation_duration_seconds": r["evaluation_duration_seconds"],
                    "metrics": r["metrics"],
                    "proxy_statistics": proxy_stats,
                    "vllm_metrics": vllm_metrics,
                })

            grouped_results.append({
                "task_name": task_name,
                "model_name": model_name,
                "inference_parameters": first_run["inference_parameters"],
                "num_repetitions": len(runs),
                "aggregate_stats": compute_aggregate_stats(runs),
                "runs": run_data
            })
        return grouped_results

    # Main execution
    session_dir = Path(artifacts_pvc_mount_path) / "evaluation-artifacts" / "sessions" / session_id
    results_dir = session_dir / "results"
    logs_dir = session_dir / "logs"
    proxy_logs_path = logs_dir / "server_proxy_statistics.jsonl"
    vllm_metrics_path = logs_dir / "vllm_metrics.jsonl"

    if not results_dir.exists():
        print(f"No results directory found at {results_dir}")
        return

    directories = [str(results_dir)]

    json_files = scan_json_files(directories)
    print(f"Found {len(json_files)} JSON files")

    # Parse proxy logs
    task_seed_proxy_stats, logs_by_task = parse_proxy_logs(proxy_logs_path)
    print(f"Parsed proxy statistics for {len(task_seed_proxy_stats)} task/seed combinations")

    # Aggregate proxy stats by task
    task_proxy_aggregates = aggregate_proxy_stats_by_task(logs_by_task, task_seed_proxy_stats)
    print(f"Aggregated proxy statistics for {len(task_proxy_aggregates)} tasks")

    # Parse vLLM metrics
    task_seed_vllm_metrics = parse_vllm_metrics(vllm_metrics_path)
    print(f"Parsed vLLM metrics for {len(task_seed_vllm_metrics)} task/seed combinations")

    # Aggregate vLLM metrics by task
    task_vllm_aggregates = aggregate_vllm_metrics_by_task(task_seed_vllm_metrics)
    print(f"Aggregated vLLM metrics for {len(task_vllm_aggregates)} tasks")

    all_results = parse_json_files(json_files)
    grouped_results = group_results(all_results, task_seed_proxy_stats, task_seed_vllm_metrics)

    # Add task-level aggregates to results
    for result in grouped_results:
        task_name = result["task_name"]
        if task_name in task_proxy_aggregates:
            result["proxy_statistics_aggregate"] = task_proxy_aggregates[task_name]
        if task_name in task_vllm_aggregates:
            result["vllm_metrics_aggregate"] = task_vllm_aggregates[task_name]

    unique_tasks = sorted(set(r["task_name"] for r in all_results))
    unique_models = sorted(set(r["model_name"] for r in all_results if r.get("model_name")))

    output_data = {
        "results": grouped_results,
        "metadata": {
            "parser_version": "1.0",
            "parse_datetime": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "model_id": model_id,
            "total_results_parsed": len(all_results),
            "total_files_parsed": len(json_files),
            "unique_tasks": unique_tasks,
            "unique_models": unique_models,
            "proxy_logs_parsed": len(task_seed_proxy_stats) > 0,
            "vllm_metrics_parsed": len(task_seed_vllm_metrics) > 0,
        }
    }

    output_dir = session_dir / "collated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "collated_results.json"

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Collated results written to: {output_path}")
    print(f"Total results: {len(grouped_results)}")
