from kfp import dsl

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["lm-eval[api]"]
)
def evaluate_model(
    service_url: str,
    tasks: str,
    session_id: str,
    reasoning_parser: str,
    model_path: str = "Qwen/Qwen3-8B",
    save_path: str = "/tier2/evaluations",
    save_prefix: str = "baseline",
) -> None:
    import os
    import subprocess
    from pathlib import Path

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
    session_dir = Path(save_path) / "sessions" / session_id / save_prefix
    session_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = session_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    results_dir = session_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Change to session directory
    os.chdir(session_dir)

    # Ensure service_url uses the correct format for base_url
    base_url = service_url.rstrip("/") + "/v1"

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

                # Build lm_eval command
                cmd = [
                    "python", "-m", "lm_eval",
                    "--model", "local-chat-completions",
                    "--tasks", task_tag,
                    "--model_args", (
                        f"model={model_path},"
                        f"max_length={max_length},"
                        f"base_url={base_url}/chat/completions,"
                        f"num_concurrent=128,"
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