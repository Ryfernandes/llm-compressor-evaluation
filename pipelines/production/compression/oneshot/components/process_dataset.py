from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["datasets", "transformers", "jinja2"]
)
def process_dataset(
    model_id: str,
    dataset_id: str,
    dataset_split: str,
    num_calibration_samples: int,
    max_sequence_length: int,
    session_id: str,
    models_mount_path: str = "/models",
    datasets_mount_path: str = "/datasets",
) -> None:
    """
    Load a calibration dataset, apply the model's chat template, tokenize,
    and save to disk for the compression step.

    Follows the dataset preparation pattern from:
    https://github.com/vllm-project/llm-compressor/blob/main/examples/awq/fp8_block_llama_example.py
    """

    import warnings
    from pathlib import Path
    from datasets import load_dataset, DatasetDict
    from transformers import AutoTokenizer

    if not dataset_id:
        warnings.warn("No dataset_id provided. Skipping dataset preparation.")
        return

    print(f"Loading tokenizer for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if dataset_split:
        print(f"Loading dataset {dataset_id} (split={dataset_split}, samples={num_calibration_samples})")
        print(f"Using dataset cache directory: {datasets_mount_path}")
        ds = load_dataset(dataset_id, split=f"{dataset_split}[:{num_calibration_samples}]", cache_dir=datasets_mount_path)
    else:
        warnings.warn(f"No dataset_split provided for dataset {dataset_id}. Attempting to infer split.")
        print(f"Loading dataset {dataset_id} (no split specified)")
        print(f"Using dataset cache directory: {datasets_mount_path}")
        ds = load_dataset(dataset_id, cache_dir=datasets_mount_path)

        if isinstance(ds, DatasetDict):
            COMMON_SPLITS = ["train", "validation", "test"]
            selected_split = None
            for candidate in COMMON_SPLITS:
                for available in ds.keys():
                    if available.startswith(candidate):
                        selected_split = available
                        break
                if selected_split:
                    break

            if not selected_split:
                selected_split = list(ds.keys())[0]

            warnings.warn(f"Inferred dataset split: '{selected_split}'")
            ds = ds[selected_split]

        ds = ds.select(range(min(num_calibration_samples, len(ds))))

    ds = ds.shuffle(seed=42)
    print(f"Dataset loaded and shuffled: {len(ds)} samples")

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)
    print("Chat template applied")

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    print(f"Dataset tokenized (max_sequence_length={max_sequence_length})")

    save_path = Path(models_mount_path) / "sessions" / session_id / "dataset"
    ds.save_to_disk(str(save_path))
    print(f"Dataset saved to: {save_path}")
