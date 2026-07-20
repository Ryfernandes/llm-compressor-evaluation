from kfp import dsl


@dsl.component(base_image="python:3.12")
def collate_results(
    session_id: str,
    served_model_name: str,
    evaluation_config_name: str,
    local_model: bool = False,
    artifacts_pvc_mount_path: str = "/artifacts",
    configs_pvc_mount_path: str = "/configs",
):
    """Collate evaluation results from model evaluation runs."""
    import json
    import os
    import statistics
    from collections import defaultdict
    from datetime import datetime, timezone
    from pathlib import Path

    # Metrics registries split by framework
    # LM-Eval: exact key matching
    LMEVAL_METRICS = {
        "gsm8k": ["exact_match,strict-match"],
        "gsm8k_platinum_cot_llama": ["exact_match,strict-match"],
        "mmlu_cot_llama": ["exact_match,strict_match"],
        "mmlu_pro_chat": ["exact_match,custom-extract"],
        "ifeval": ["inst_level_strict_acc,none"],
        "mrcr": ["score_gt_16k_le_32k", "AUC"],
    }

    # LightEval: prefix matching (part before @ symbol)
    # The suffix after @ varies based on evaluation settings
    LIGHTEVAL_METRICS = {
        "math_500|0": ["pass@"],
        "aime25|0": ["pass@"],
        "gpqa:diamond|0": ["gpqa_pass@"],
        "lcb:codegeneration_v6|0": ["codegen_pass@"],
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

    def get_base_task_tag(task_id):
        """
        Extract base task tag from lighteval task ID.
        LightEval adds |N suffix to task IDs (e.g., 'math_500|0'),
        but logging uses the base tag ('math_500').
        """
        if "|" in task_id:
            return task_id.split("|")[0]
        return task_id

    def extract_lmeval_metrics(task_name, task_results):
        """Extract metrics from LM-Eval format using exact key matching."""
        metrics = {}
        if task_name in LMEVAL_METRICS:
            for metric_key in LMEVAL_METRICS[task_name]:
                value = task_results.get(metric_key)
                stderr_key = f"{metric_key}_stderr"
                stderr = task_results.get(stderr_key)
                if stderr == "N/A":
                    stderr = None
                if value is not None:
                    metrics[metric_key] = {"value": value, "stderr": stderr}
        return metrics

    def parse_timestamp_from_filename(filename):
        """
        Extract ISO timestamp from filename if present.
        Both lm-eval and lighteval include ISO timestamps in filenames.
        Example: results_2026-07-06T08-03-46.941709.json
        Returns UNIX timestamp or None if parsing fails.
        """
        import re
        # Pattern: YYYY-MM-DDTHH-MM-SS.microseconds
        pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)'
        match = re.search(pattern, filename)
        if match:
            extracted = match.group(1)
            # Convert from: 2026-07-06T08-03-46.941709
            # To ISO format: 2026-07-06T08:03:46.941709
            iso_string = extracted[:10] + 'T' + extracted[11:13] + ':' + extracted[14:16] + ':' + extracted[17:]
            try:
                dt = datetime.fromisoformat(iso_string).replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except (ValueError, AttributeError):
                pass
        return None

    def extract_lighteval_metrics(task_name, task_results):
        """Extract metrics from LightEval format using prefix matching.

        LightEval metric keys have dynamic suffixes after @ (e.g., 'pass@1:16', 'codegen_pass@1:16')
        that depend on evaluation settings. We match based on the prefix before @.
        """
        metrics = {}
        if task_name in LIGHTEVAL_METRICS:
            for metric_prefix in LIGHTEVAL_METRICS[task_name]:
                # Find all keys that start with this prefix and don't end with _stderr
                for key in task_results.keys():
                    if key.startswith(metric_prefix) and not key.endswith("_stderr"):
                        value = task_results.get(key)
                        stderr_key = f"{key}_stderr"
                        stderr = task_results.get(stderr_key)
                        if stderr == "N/A":
                            stderr = None
                        if value is not None:
                            metrics[key] = {"value": value, "stderr": stderr}
        return metrics

    def extract_lmeval_result(data, task_name, source_filename, task_concurrency_map, task_limit_map, harness_metadata):
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

        # Get harness info for this task
        task_harness_info = harness_metadata.get(task_name, {})

        return {
            "task_name": task_name,
            "model_name": data.get("model_name"),
            "source_filename": source_filename,
            "evaluation_datetime": timestamp,
            "evaluation_datetime_iso": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat() if timestamp else None,
            "evaluation_duration_seconds": duration,
            "metrics": metrics,
            "harness": task_harness_info.get("harness"),
            "harness_version": task_harness_info.get("version"),
            "harness_details": task_harness_info.get("harness_details"),
            "inference_parameters": {
                "do_sample": gen_kwargs.get("do_sample"),
                "temperature": gen_kwargs.get("temperature"),
                "top_p": gen_kwargs.get("top_p"),
                "top_k": gen_kwargs.get("top_k"),
                "max_gen_toks": gen_kwargs.get("max_gen_toks"),
                "seed": gen_kwargs.get("seed"),
                "concurrency": task_concurrency_map.get(task_name),
                "limit": task_limit_map.get(task_name),
            },
        }

    def extract_lighteval_result(data, task_id, source_filename, task_concurrency_map, task_limit_map, harness_metadata):
        """Extract single task result from LightEval."""
        task_results = data["results"][task_id]
        metrics = extract_lighteval_metrics(task_id, task_results)

        config_general = data.get("config_general", {})
        gen_params = config_general.get("model_config", {}).get("generation_parameters", {})

        # LightEval's start_time is monotonic (system uptime), not UNIX epoch
        # Parse timestamp from filename instead (both frameworks include ISO timestamp in filename)
        timestamp = parse_timestamp_from_filename(source_filename)

        duration = config_general.get("total_evaluation_time_secondes")  # Note: typo in lighteval
        if isinstance(duration, str):
            try:
                duration = float(duration)
            except (ValueError, TypeError):
                duration = None

        # LightEval task IDs have |N suffix (e.g., 'math_500|0'),
        # but logging/config uses base tag (e.g., 'math_500')
        base_task_tag = get_base_task_tag(task_id)

        # Get harness info for this task (using base tag)
        task_harness_info = harness_metadata.get(base_task_tag, {})

        temperature = gen_params.get("temperature", 0)

        return {
            "task_name": task_id,
            "model_name": config_general.get("model_name"),
            "source_filename": source_filename,
            "evaluation_datetime": timestamp,
            "evaluation_datetime_iso": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat() if timestamp else None,
            "evaluation_duration_seconds": duration,
            "metrics": metrics,
            "harness": task_harness_info.get("harness"),
            "harness_version": task_harness_info.get("version"),
            "harness_details": task_harness_info.get("harness_details"),
            "inference_parameters": {
                "do_sample": temperature is not None and temperature > 0,  # Inferred
                "temperature": temperature,
                "top_p": gen_params.get("top_p"),
                "top_k": gen_params.get("top_k"),
                "max_gen_toks": gen_params.get("max_new_tokens"),  # Different name
                "seed": gen_params.get("seed"),
                "concurrency": task_concurrency_map.get(base_task_tag),  # Use base tag for lookups
                "limit": task_limit_map.get(base_task_tag),  # Use base tag for lookups
            },
        }

    def parse_json_files(json_files, task_concurrency_map, task_limit_map, harness_metadata):
        """Parse all JSON files and extract results."""
        all_results = []
        for json_file_path, source_dir in json_files:
            try:
                with open(json_file_path, "r") as f:
                    json_data = json.load(f)

                framework = detect_framework(json_data)
                if framework is None:
                    print(f"Skipping unknown framework file: {json_file_path}")
                    continue

                source_filename = os.path.basename(json_file_path)

                if framework == "lmeval":
                    for task_name in json_data.get("results", {}).keys():
                        if task_name in LMEVAL_METRICS:
                            result = extract_lmeval_result(json_data, task_name, source_filename, task_concurrency_map, task_limit_map, harness_metadata)
                            result["source_directory"] = source_dir
                            all_results.append(result)
                        else:
                            print(f"Skipping unknown task '{task_name}' in {json_file_path}")
                elif framework == "lighteval":
                    for task_id in json_data.get("results", {}).keys():
                        if task_id == "all":
                            continue
                        if task_id in LIGHTEVAL_METRICS:
                            result = extract_lighteval_result(json_data, task_id, source_filename, task_concurrency_map, task_limit_map, harness_metadata)
                            result["source_directory"] = source_dir
                            all_results.append(result)
                        else:
                            print(f"Skipping unknown task '{task_id}' in {json_file_path}")
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

    def group_results(all_results, task_seed_proxy_stats, task_seed_vllm_metrics, task_seed_log_stats):
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

                # For lighteval tasks, strip |N suffix to match logged task_id
                # Logging uses base tag (e.g., 'math_500'), but JSON has full ID (e.g., 'math_500|0')
                lookup_task_name = get_base_task_tag(task_name)

                # Find matching proxy stats for this task/seed
                proxy_stats = None
                if seed is not None and (lookup_task_name, seed) in task_seed_proxy_stats:
                    proxy_stats = task_seed_proxy_stats[(lookup_task_name, seed)]

                # Find matching vLLM metrics for this task/seed
                vllm_metrics = None
                if seed is not None and (lookup_task_name, seed) in task_seed_vllm_metrics:
                    vllm_metrics = task_seed_vllm_metrics[(lookup_task_name, seed)]

                # Find matching log statistics for this task/seed
                log_stats = None
                if seed is not None and (lookup_task_name, seed) in task_seed_log_stats:
                    log_stats = task_seed_log_stats[(lookup_task_name, seed)]

                run_data.append({
                    "source_filename": r["source_filename"],
                    "evaluation_datetime": r["evaluation_datetime"],
                    "evaluation_datetime_iso": r["evaluation_datetime_iso"],
                    "evaluation_duration_seconds": r["evaluation_duration_seconds"],
                    "metrics": r["metrics"],
                    "proxy_statistics": proxy_stats,
                    "vllm_metrics": vllm_metrics,
                    "log_statistics": log_stats,
                })

            grouped_results.append({
                "task_name": task_name,
                "model_name": model_name,
                "harness": first_run.get("harness"),
                "harness_version": first_run.get("harness_version"),
                "harness_details": first_run.get("harness_details"),
                "inference_parameters": first_run["inference_parameters"],
                "num_repetitions": len(runs),
                "aggregate_stats": compute_aggregate_stats(runs),
                "runs": run_data
            })
        return grouped_results

    def parse_harness_metadata(logs_path):
        """Parse harness metadata JSON file containing harness name and version for each task."""
        harness_metadata_path = logs_path.parent / "harness_metadata.json"

        if not harness_metadata_path.exists():
            print(f"No harness metadata found at {harness_metadata_path}")
            return {}

        try:
            with open(harness_metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"ERROR reading harness metadata: {e}")
            return {}

    def parse_vllm_log_statistics(logs_path):
        """Parse vLLM log statistics JSON file."""
        if not logs_path.exists():
            print(f"No vLLM log statistics found at {logs_path}")
            return {}, {}

        try:
            with open(logs_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"ERROR reading vLLM log statistics: {e}")
            return {}, {}

        # Extract startup statistics
        startup_stats = data.get("start", {})

        # Parse task-level data: tasks -> task_id -> seeds -> [samples]
        task_seed_samples = {}
        tasks_data = data.get("tasks", {})

        for task_id, task_data in tasks_data.items():
            seeds_data = task_data.get("seeds", {})
            for seed_str, samples in seeds_data.items():
                seed = int(seed_str)
                task_seed_samples[(task_id, seed)] = samples

        return startup_stats, task_seed_samples

    def compute_log_stats(samples):
        """Compute min/max/mean/p95/p99 for KV cache and throughput metrics from log samples."""
        if not samples:
            return None

        # Extract all numeric fields
        prompt_throughput = [s["prompt_throughput"] for s in samples if "prompt_throughput" in s]
        generation_throughput = [s["generation_throughput"] for s in samples if "generation_throughput" in s]
        running_reqs = [s["running_reqs"] for s in samples if "running_reqs" in s]
        waiting_reqs = [s["waiting_reqs"] for s in samples if "waiting_reqs" in s]
        kv_cache_usage_pct = [s["kv_cache_usage_pct"] for s in samples if "kv_cache_usage_pct" in s]
        prefix_cache_hit_rate = [s["prefix_cache_hit_rate"] for s in samples if "prefix_cache_hit_rate" in s]

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

        return {
            "prompt_throughput": stats_dict(prompt_throughput),
            "generation_throughput": stats_dict(generation_throughput),
            "running_reqs": stats_dict(running_reqs),
            "waiting_reqs": stats_dict(waiting_reqs),
            "kv_cache_usage_pct": stats_dict(kv_cache_usage_pct),
            "prefix_cache_hit_rate": stats_dict(prefix_cache_hit_rate),
            "num_samples": len(samples),
        }

    def compute_task_seed_log_stats(task_seed_samples):
        """Compute statistics for each (task, seed) combination."""
        task_seed_stats = {}
        for (task_id, seed), samples in task_seed_samples.items():
            stats = compute_log_stats(samples)
            task_seed_stats[(task_id, seed)] = stats
        return task_seed_stats

    def aggregate_log_stats_by_task(task_seed_samples):
        """Aggregate log statistics at the task level across all seeds."""
        # Group samples by task_id
        samples_by_task = defaultdict(list)
        for (task_id, seed), samples in task_seed_samples.items():
            samples_by_task[task_id].extend(samples)

        # Aggregate per task
        task_aggregates = {}
        for task_id, all_samples in samples_by_task.items():
            stats = compute_log_stats(all_samples)

            # Add num_seeds count
            unique_seeds = set()
            for (tid, seed), _ in task_seed_samples.items():
                if tid == task_id:
                    unique_seeds.add(seed)
            if stats:
                stats["num_seeds"] = len(unique_seeds)

            task_aggregates[task_id] = stats

        return task_aggregates

    # Main execution
    session_dir = Path(artifacts_pvc_mount_path) / "evaluation-artifacts" / "sessions" / session_id
    results_dir = session_dir / "results"
    logs_dir = session_dir / "logs"
    proxy_logs_path = logs_dir / "server_proxy_statistics.jsonl"
    vllm_metrics_path = logs_dir / "vllm_metrics.jsonl"
    vllm_log_stats_path = logs_dir / "vllm_log_statistics.json"

    # Load evaluation config to get task concurrency mapping
    evaluation_config_path = Path(configs_pvc_mount_path) / "evaluation" / evaluation_config_name
    with open(evaluation_config_path, 'r') as f:
        evaluation_config = json.load(f)

    # Create task_name -> concurrency and limit mappings
    task_concurrency = {}
    task_limit = {}
    for task in evaluation_config.get("tasks", []):
        task_tag = task.get("tag")
        concurrency = task.get("concurrency")
        limit = task.get("limit")

        if task_tag and concurrency:
            task_concurrency[task_tag] = concurrency

        if task_tag:
            # limit: if 0 or not present, set to None
            if limit == 0 or limit is None:
                task_limit[task_tag] = None
            else:
                task_limit[task_tag] = limit

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

    # Parse vLLM log statistics (startup + KV cache utilization)
    startup_stats, task_seed_log_samples = parse_vllm_log_statistics(vllm_log_stats_path)
    print(f"Parsed vLLM log statistics for {len(task_seed_log_samples)} task/seed combinations")

    # Compute per-(task, seed) log statistics
    task_seed_log_stats = compute_task_seed_log_stats(task_seed_log_samples)
    print(f"Computed log statistics for {len(task_seed_log_stats)} task/seed combinations")

    # Aggregate log statistics by task
    task_log_aggregates = aggregate_log_stats_by_task(task_seed_log_samples)
    print(f"Aggregated log statistics for {len(task_log_aggregates)} tasks")

    # Parse harness metadata (harness name and version for each task)
    harness_metadata = parse_harness_metadata(vllm_log_stats_path)
    print(f"Parsed harness metadata for {len(harness_metadata)} tasks")

    all_results = parse_json_files(json_files, task_concurrency, task_limit, harness_metadata)
    grouped_results = group_results(all_results, task_seed_proxy_stats, task_seed_vllm_metrics, task_seed_log_stats)

    # Add task-level aggregates to results
    for result in grouped_results:
        task_name = result["task_name"]
        # For lighteval tasks, strip |N suffix to match logged task_id
        lookup_task_name = get_base_task_tag(task_name)

        if lookup_task_name in task_proxy_aggregates:
            result["proxy_statistics_aggregate"] = task_proxy_aggregates[lookup_task_name]
        if lookup_task_name in task_vllm_aggregates:
            result["vllm_metrics_aggregate"] = task_vllm_aggregates[lookup_task_name]
        if lookup_task_name in task_log_aggregates:
            result["log_statistics_aggregate"] = task_log_aggregates[lookup_task_name]

    unique_tasks = sorted(set(r["task_name"] for r in all_results))
    unique_models = sorted(set(r["model_name"] for r in all_results if r.get("model_name")))

    metadata = {
        "parser_version": "1.0",
        "parse_datetime": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "evaluated_from_local_checkpoint": local_model,
        "total_results_parsed": len(all_results),
        "total_files_parsed": len(json_files),
        "unique_tasks": unique_tasks,
        "unique_models": unique_models,
        "proxy_logs_parsed": len(task_seed_proxy_stats) > 0,
        "vllm_metrics_parsed": len(task_seed_vllm_metrics) > 0,
        "log_statistics_parsed": len(task_seed_log_stats) > 0,
    }

    if local_model:
        metadata["model_path"] = served_model_name
    else:
        metadata["model_id"] = served_model_name

    output_data = {
        "server": startup_stats if startup_stats else None,
        "results": grouped_results,
        "metadata": metadata,
    }

    output_dir = session_dir / "collated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "collated_results.json"

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Collated results written to: {output_path}")
    print(f"Total results: {len(grouped_results)}")
