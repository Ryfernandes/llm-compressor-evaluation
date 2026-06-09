import json
import os
from collections import defaultdict
from datetime import datetime, timezone
import statistics

from scanner import scan_json_files
from detector import detect_framework


class Parser:
    """
    Parses evaluation results from LM-Eval and LightEval tasks
    Extracts relevant information and adds references to source files in a JSON output
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def parse(self, dirs: list[str]):
        json_files = scan_json_files(dirs)

        all_results = []
        files_parsed = 0

        for json_file_path in json_files:
            try:
                with open(json_file_path, "r") as f:
                    json_data = json.load(f)

                adapter = detect_framework(json_data)
                source_filename = os.path.basename(json_file_path)
                results = adapter.extract_results(json_data, source_filename)

                all_results.extend(results)
                files_parsed += 1

            except Exception as e:
                print(f"  ERROR parsing {json_file_path}: {e}")
                continue

        # Group results by (task_name, model_name)
        grouped_results = self._group_results(all_results)

        unique_tasks = sorted(set(r["task_name"] for r in all_results))
        unique_models = sorted(set(r["model_name"] for r in all_results if r.get("model_name")))

        metadata = {
            "parser_version": "1.0",
            "parse_datetime": datetime.now(timezone.utc).isoformat(),
            "total_results_parsed": len(all_results),
            "total_files_parsed": files_parsed,
            "unique_tasks": unique_tasks,
            "unique_models": unique_models
        }

        output_data = {
            "results": grouped_results,
            "metadata": metadata,
        }

        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_filename = f"parsed_results_{timestamp}.json"
        output_path = os.path.join(self.output_dir, output_filename)

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        return output_path

    def _group_results(self, all_results: list[dict]) -> list[dict]:
        """
        Group results by (task_name, model_name) and compute aggregate statistics.

        Returns a list of grouped results where each entry contains:
        - Shared fields (task_name, model_name, inference_parameters)
        - num_repetitions: count of runs
        - aggregate_stats: mean and std for duration and all metrics
        - runs: list of individual run data (metrics, duration, datetime, source)
        """
        # Group by (task_name, model_name)
        groups = defaultdict(list)
        for result in all_results:
            key = (result["task_name"], result["model_name"])
            groups[key].append(result)

        grouped_results = []
        for (task_name, model_name), runs in groups.items():
            # Shared fields (from first run, assumed identical across runs)
            first_run = runs[0]

            # Extract individual run data
            run_data = []
            for run in runs:
                run_data.append({
                    "source_filename": run["source_filename"],
                    "evaluation_datetime": run["evaluation_datetime"],
                    "evaluation_datetime_iso": run["evaluation_datetime_iso"],
                    "evaluation_duration_seconds": run["evaluation_duration_seconds"],
                    "metrics": run["metrics"]
                })

            # Compute aggregate statistics
            aggregate_stats = self._compute_aggregate_stats(runs)

            # Build grouped result
            grouped_result = {
                "task_name": task_name,
                "model_name": model_name,
                "inference_parameters": first_run["inference_parameters"],
                "num_repetitions": len(runs),
                "aggregate_stats": aggregate_stats,
                "runs": run_data
            }

            grouped_results.append(grouped_result)

        return grouped_results

    def _compute_aggregate_stats(self, runs: list[dict]) -> dict:
        """
        Compute aggregate statistics across multiple runs.

        Returns dict with:
        - evaluation_duration_seconds: {mean, std}
        - metrics: {metric_name: {value: {mean, std}, stderr: {mean, std}}}
        """
        stats = {}

        # Duration statistics
        durations = [r["evaluation_duration_seconds"] for r in runs if r["evaluation_duration_seconds"] is not None]
        if durations:
            stats["evaluation_duration_seconds"] = {
                "mean": statistics.mean(durations),
                "std": statistics.stdev(durations) if len(durations) > 1 else 0.0
            }
        else:
            stats["evaluation_duration_seconds"] = {"mean": None, "std": None}

        # Metrics statistics
        # First, collect all metric names
        all_metric_names = set()
        for run in runs:
            all_metric_names.update(run["metrics"].keys())

        metrics_stats = {}
        for metric_name in all_metric_names:
            # Collect values and stderrs for this metric across runs
            values = []
            stderrs = []

            for run in runs:
                if metric_name in run["metrics"]:
                    metric_data = run["metrics"][metric_name]
                    if metric_data["value"] is not None:
                        values.append(metric_data["value"])
                    if metric_data["stderr"] is not None:
                        stderrs.append(metric_data["stderr"])

            metric_stats = {}

            # Value statistics
            if values:
                metric_stats["value"] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0
                }
            else:
                metric_stats["value"] = {"mean": None, "std": None}

            # Stderr statistics
            if stderrs:
                metric_stats["stderr"] = {
                    "mean": statistics.mean(stderrs),
                    "std": statistics.stdev(stderrs) if len(stderrs) > 1 else 0.0
                }
            else:
                metric_stats["stderr"] = {"mean": None, "std": None}

            metrics_stats[metric_name] = metric_stats

        stats["metrics"] = metrics_stats

        return stats