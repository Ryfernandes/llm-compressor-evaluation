from kfp import dsl


@dsl.component(base_image="python:3.12")
def collate_results(
    session_id: str,
    save_path: str = "/tier2/evaluations",
):
    """Collate evaluation results from baseline and compressed model runs."""
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

                # Determine model type from source directory
                model_type = "compressed" if "compressed" in source_dir else "baseline"

                for task_name in json_data.get("results", {}).keys():
                    if task_name in METRICS_REGISTRY:
                        result = extract_lmeval_result(json_data, task_name, source_filename)
                        result["model_type"] = model_type
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

    def group_results(all_results):
        """Group results by (task_name, model_name, model_type)."""
        groups = defaultdict(list)
        for result in all_results:
            # model_type was already set in parse_json_files
            model_type = result["model_type"]
            key = (result["task_name"], result["model_name"], model_type)
            groups[key].append(result)

        grouped_results = []
        for (task_name, model_name, model_type), runs in groups.items():
            first_run = runs[0]
            run_data = [{
                "source_filename": r["source_filename"],
                "evaluation_datetime": r["evaluation_datetime"],
                "evaluation_datetime_iso": r["evaluation_datetime_iso"],
                "evaluation_duration_seconds": r["evaluation_duration_seconds"],
                "metrics": r["metrics"]
            } for r in runs]

            grouped_results.append({
                "task_name": task_name,
                "model_name": model_name,
                "model_type": model_type,
                "inference_parameters": first_run["inference_parameters"],
                "num_repetitions": len(runs),
                "aggregate_stats": compute_aggregate_stats(runs),
                "runs": run_data
            })
        return grouped_results

    # Main execution
    session_dir = Path(save_path) / "sessions" / session_id
    baseline_dir = session_dir / "baseline" / "results"
    compressed_dir = session_dir / "compressed" / "results"

    directories = []
    if baseline_dir.exists():
        directories.append(str(baseline_dir))
    if compressed_dir.exists():
        directories.append(str(compressed_dir))

    if not directories:
        print("No results directories found")
        return

    json_files = scan_json_files(directories)
    print(f"Found {len(json_files)} JSON files")

    all_results = parse_json_files(json_files)
    grouped_results = group_results(all_results)

    # Separate by model type
    baseline_results = [r for r in grouped_results if r["model_type"] == "baseline"]
    compressed_results = [r for r in grouped_results if r["model_type"] == "compressed"]

    unique_tasks = sorted(set(r["task_name"] for r in all_results))
    unique_models = sorted(set(r["model_name"] for r in all_results if r.get("model_name")))

    output_data = {
        "baseline_results": baseline_results,
        "compressed_results": compressed_results,
        "metadata": {
            "parser_version": "1.0",
            "parse_datetime": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "total_results_parsed": len(all_results),
            "total_files_parsed": len(json_files),
            "unique_tasks": unique_tasks,
            "unique_models": unique_models
        }
    }

    output_dir = session_dir / "collated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "collated_results.json"

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Collated results written to: {output_path}")
    print(f"Baseline results: {len(baseline_results)}")
    print(f"Compressed results: {len(compressed_results)}")
