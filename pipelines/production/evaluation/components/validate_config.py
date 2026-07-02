from kfp import dsl

@dsl.component(
    base_image="python:3.12"
)
def validate_config(
    config_filename: str,
    configs_pvc_mount_path: str = "/configs",
) -> None:
    import json
    from pathlib import Path

    # Load configuration file
    config_path = Path(configs_pvc_mount_path) / config_filename
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Validating config file: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate model configuration
    model_config = config.get("model")
    if not model_config:
        raise ValueError("Config validation failed: 'model' section is required")

    required_model_fields = ["max_model_len", "temperature", "top_p", "top_k"]
    missing_model_fields = [field for field in required_model_fields if field not in model_config]

    if missing_model_fields:
        raise ValueError(f"Config validation failed: 'model' section missing required fields: {missing_model_fields}")

    max_model_len = model_config["max_model_len"]
    print(f"Model config validated: max_model_len={max_model_len}, temperature={model_config['temperature']}, top_p={model_config['top_p']}, top_k={model_config['top_k']}")

    # Validate tasks
    tasks = config.get("tasks")
    if not tasks:
        raise ValueError("Config validation failed: 'tasks' array is required and must not be empty")

    if not isinstance(tasks, list):
        raise ValueError("Config validation failed: 'tasks' must be an array")

    if len(tasks) == 0:
        raise ValueError("Config validation failed: 'tasks' array must contain at least one task")

    required_task_fields = ["harness", "tag", "shots", "reps", "concurrency", "max_tokens"]

    for i, task in enumerate(tasks):
        task_id = task.get("tag", f"task[{i}]")

        # Check required fields
        missing_fields = [field for field in required_task_fields if field not in task]
        if missing_fields:
            raise ValueError(f"Config validation failed: Task '{task_id}' missing required fields: {missing_fields}")

        # Validate max_tokens against max_model_len
        max_tokens = task["max_tokens"]
        if max_tokens > max_model_len:
            raise ValueError(f"Config validation failed: Task '{task_id}' has max_tokens ({max_tokens}) exceeding model's max_model_len ({max_model_len})")

        print(f"Task '{task_id}' validated: harness={task['harness']}, shots={task['shots']}, reps={task['reps']}, concurrency={task['concurrency']}, max_tokens={max_tokens}")

    print(f"\nConfig validation passed: {len(tasks)} tasks validated successfully")
