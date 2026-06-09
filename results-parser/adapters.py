from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from registry import METRICS_REGISTRY


class ResultAdapter(ABC):
    """Base adapter interface"""

    @abstractmethod
    def extract_results(self, json_data: dict, source_filename: str) -> list[dict]:
        """
        Parses data to extract task results in a consistent format
        """
        pass


class LMEvalAdapter(ResultAdapter):
    """Adapter for LM-Eval results"""

    def extract_results(self, json_data: dict, source_filename: str) -> list[dict]:
        results = []

        for task_name in json_data.get("results", {}).keys():
            if task_name not in METRICS_REGISTRY:
                continue

            result = self._extract_task_result(json_data, task_name, source_filename)
            if result:
                results.append(result)

        return results

    def _extract_task_result(
        self, data: dict, task_name: str, source_filename: str
    ) -> dict | None:
        """Parsing for a single task"""
        task_results = data["results"][task_name]

        metrics = self._extract_metrics(task_name, task_results)
        inference_params = self._extract_inference_params(data)

        timestamp = data.get("date")
        duration = data.get("total_evaluation_time_seconds")

        if isinstance(duration, str):
            try:
                duration = float(duration)
            except (ValueError, TypeError):
                duration = None

        result = {
            "task_name": task_name,
            "model_name": data.get("model_name"),
            "source_filename": source_filename,
            "evaluation_datetime": timestamp,
            "evaluation_datetime_iso": (
                datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
                if timestamp
                else None
            ),
            "evaluation_duration_seconds": duration,
            "metrics": metrics,
            "inference_parameters": inference_params,
        }

        return result

    def _extract_metrics(self, task_name: str, task_results: dict) -> dict:
        metrics = {}

        if task_name in METRICS_REGISTRY:
            # Use registry-specified metrics
            for metric_key in METRICS_REGISTRY[task_name]:
                value = task_results.get(metric_key)
                stderr = self._find_stderr(metric_key, task_results)

                if value is not None:
                    metrics[metric_key] = {"value": value, "stderr": stderr}
        else:
            metadata_fields = {"name", "alias", "sample_len"}

            for key, value in task_results.items():
                if key in metadata_fields or "_stderr" in key:
                    continue

                if isinstance(value, (int, float)):
                    stderr = self._find_stderr(key, task_results)
                    metrics[key] = {"value": value, "stderr": stderr}

        return metrics

    def _find_stderr(self, metric_key: str, task_results: dict) -> float | None:
        # Try inserting _stderr before each comma
        comma_indices = [i for i, c in enumerate(metric_key) if c == ","]

        for i in comma_indices:
            stderr_key = metric_key[:i] + "_stderr" + metric_key[i:]
            stderr = task_results.get(stderr_key)
            if stderr is not None:
                return None if stderr == "N/A" else stderr

        # Try appending _stderr at the end
        stderr_key = f"{metric_key}_stderr"
        stderr = task_results.get(stderr_key)
        if stderr is not None:
            return None if stderr == "N/A" else stderr

        return None

    def _extract_inference_params(self, data: dict) -> dict:
        gen_kwargs = data.get("config", {}).get("gen_kwargs", {})

        return {
            "do_sample": gen_kwargs.get("do_sample"),
            "temperature": gen_kwargs.get("temperature"),
            "top_p": gen_kwargs.get("top_p"),
            "top_k": gen_kwargs.get("top_k"),
            "max_gen_toks": gen_kwargs.get("max_gen_toks"),
            "seed": gen_kwargs.get("seed"),
        }


class LightEvalAdapter(ResultAdapter):
    """Adapter for LightEval results"""

    def extract_results(self, json_data: dict, source_filename: str) -> list[dict]:
        results = []

        for task_id in json_data.get("results", {}).keys():
            if task_id == "all" or task_id not in METRICS_REGISTRY:
                continue

            result = self._extract_task_result(json_data, task_id, source_filename)
            if result:
                results.append(result)

        return results

    def _extract_task_result(
        self, data: dict, task_id: str, source_filename: str
    ) -> dict | None:
        """Parsing for a single task"""
        task_results = data["results"][task_id]

        metrics = self._extract_metrics(task_id, task_results)
        inference_params = self._extract_inference_params(data)

        config_general = data.get("config_general", {})
        timestamp = config_general.get("start_time")
        duration = config_general.get("total_evaluation_time_secondes")  # Note: typo in lighteval

        if isinstance(duration, str):
            try:
                duration = float(duration)
            except (ValueError, TypeError):
                duration = None

        task_name = task_id

        result = {
            "task_name": task_name,
            "model_name": config_general.get("model_name"),
            "source_filename": source_filename,
            "evaluation_datetime": timestamp,
            "evaluation_datetime_iso": (
                datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
                if timestamp
                else None
            ),
            "evaluation_duration_seconds": duration,
            "metrics": metrics,
            "inference_parameters": inference_params,
        }

        return result

    def _extract_metrics(self, task_id: str, task_results: dict) -> dict:
        metrics = {}

        if task_id in METRICS_REGISTRY:
            for metric_key in METRICS_REGISTRY[task_id]:
                value = task_results.get(metric_key)
                stderr_key = f"{metric_key}_stderr"
                stderr = task_results.get(stderr_key)

                if stderr == "N/A":
                    stderr = None

                if value is not None:
                    metrics[metric_key] = {"value": value, "stderr": stderr}
        else:
            for key, value in task_results.items():
                if "_stderr" in key:
                    continue

                if isinstance(value, (int, float)):
                    stderr_key = f"{key}_stderr"
                    stderr = task_results.get(stderr_key)

                    # Handle "N/A" values
                    if stderr == "N/A":
                        stderr = None

                    metrics[key] = {"value": value, "stderr": stderr}

        return metrics

    def _extract_inference_params(self, data: dict) -> dict:
        gen_params = (
            data.get("config_general", {})
            .get("model_config", {})
            .get("generation_parameters", {})
        )

        temperature = gen_params.get("temperature", 0)

        return {
            "do_sample": temperature is not None and temperature > 0,  # Inferred
            "temperature": temperature,
            "top_p": gen_params.get("top_p"),
            "top_k": gen_params.get("top_k"),
            "max_gen_toks": gen_params.get("max_new_tokens"),  # Different name
            "seed": gen_params.get("seed"),
        }
