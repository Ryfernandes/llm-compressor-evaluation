import lm_eval
import json
import os


def handle_non_serializable(obj):
    """Convert non-JSON-serializable objects to serializable format."""
    if callable(obj):
        return str(obj)
    try:
        import numpy as np
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
    except ImportError:
        pass
    if isinstance(obj, set):
        return list(obj)
    return str(obj)

output_filename = "./outputs/lm_eval_gsm8k_chat_template_fewshot_multiturn.json"
save_samples = False

model_name = "meta-llama/Llama-3.1-8B-Instruct"
model_args = f"pretrained={model_name},dtype=auto,gpu_memory_utilization=0.8,max_model_len=4096"

os.environ["HF_DATASETS_CACHE"] = "/home/Ryfernandes/tmp_cache"

results = lm_eval.simple_evaluate(
    model="vllm",
    model_args=model_args,
    tasks=["gsm8k"],
    num_fewshot=5,
    batch_size="auto",
    apply_chat_template=True,
    fewshot_as_multiturn=True,
)

print(lm_eval.utils.make_table(results))

results_to_save = results.copy()
if not save_samples:
    results_to_save.pop("samples", None)

os.makedirs(os.path.dirname(output_filename), exist_ok=True)
with open(output_filename, "w") as f:
    json.dump(results_to_save, f, indent=2, default=handle_non_serializable)