# Evaluation Pipeline Configuration Guide

This document describes the configuration file format for the LLM evaluation pipeline.

## Overview

The evaluation pipeline uses a JSON configuration file that specifies:
- Model serving parameters (vLLM server configuration)
- Evaluation tasks to run across different harnesses
- Sampling parameters for generation

Configuration files should be stored in the configs PVC and referenced by filename when launching the pipeline.

## Configuration Structure

```json
{
  "model": {
    // Model serving and generation parameters
  },
  "tasks": [
    // Array of evaluation tasks
  ]
}
```

## Model Configuration

The `model` section configures the vLLM server and generation parameters.

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `max_model_len` | integer | Maximum sequence length (context + generation) supported by the model. Must be ≥ largest `max_tokens` across all tasks. | `16384` |
| `temperature` | float | Sampling temperature for generation. Higher values = more random. Range: `0.0` to `2.0`. | `0.8` |
| `top_p` | float | Nucleus sampling parameter. Only tokens with cumulative probability ≤ `top_p` are considered. Range: `0.0` to `1.0`. | `0.95` |
| `top_k` | integer | Top-k sampling parameter. Only the top `k` most likely tokens are considered. | `50` |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reasoning_parser` | string | `""` | Enables reasoning mode for models that generate explicit reasoning traces (e.g., `"qwen_rstar"`). Leave empty for standard generation. |
| `tp` | integer | `1` | Tensor parallelism degree. Number of GPUs to split the model across. Must be a power of 2. |
| `dp` | integer | `1` | Data parallelism degree. Number of independent model replicas for throughput. |

### Example

```json
{
  "model": {
    "reasoning_parser": "",
    "max_model_len": 16384,
    "tp": 1,
    "dp": 1,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50
  }
}
```

## Task Configuration

The `tasks` array contains evaluation tasks to run. Each task specifies which evaluation harness to use and task-specific parameters.

### Required Fields (All Harnesses)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `harness` | string | Evaluation harness identifier. Currently supported: `"lm_eval"`. | `"lm_eval"` |
| `tag` | string | Task identifier within the harness (e.g., lm-eval task name). | `"gsm8k_platinum_cot_llama"` |
| `shots` | integer | Number of few-shot examples to include in the prompt. Range: `0` to `10`. | `5` |
| `reps` | integer | Number of evaluation repetitions with different seeds. Each rep produces independent results. | `3` |
| `concurrency` | integer | Number of concurrent inference requests. Higher values increase throughput but require more GPU memory. | `256` |
| `max_tokens` | integer | Maximum number of tokens to generate per example. Must be ≤ `model.max_model_len`. | `16000` |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_seed` | integer | `1234` | Base random seed for the first repetition. Subsequent reps use `base_seed + rep_number`. |
| `timeout` | integer | `1800` | Request timeout in seconds for each inference call. |

### Task Filtering by Harness

Each evaluation component filters tasks based on the `harness` field:
- `lm_eval_evaluation` component: processes tasks where `harness == "lm_eval"`
- Future harnesses (e.g., `lighteval`) will filter for their respective harness identifier

This allows a single config to specify tasks for multiple evaluation frameworks.

### Example

```json
{
  "tasks": [
    {
      "harness": "lm_eval",
      "tag": "gsm8k_platinum_cot_llama",
      "shots": 5,
      "reps": 3,
      "concurrency": 256,
      "max_tokens": 16000,
      "base_seed": 1000,
      "timeout": 3600
    },
    {
      "harness": "lm_eval",
      "tag": "mmlu_pro_chat",
      "shots": 5,
      "reps": 3,
      "concurrency": 128,
      "max_tokens": 8192
    }
  ]
}
```

## Complete Example

See `references/example_config.json` for a complete working example:

```json
{
  "model": {
    "reasoning_parser": "",
    "max_model_len": 16384,
    "tp": 1,
    "dp": 1,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50
  },
  "tasks": [
    {
      "harness": "lm_eval",
      "tag": "gsm8k_platinum_cot_llama",
      "shots": 5,
      "reps": 3,
      "concurrency": 256,
      "max_tokens": 16000,
      "base_seed": 1000,
      "timeout": 3600
    }
  ]
}
```

## Validation Rules

The pipeline validates the configuration file before starting expensive operations (GPU pod creation). Validation failures will cause the pipeline to fail immediately with a descriptive error message.

### Model Validation

- ✅ `model` section must exist
- ✅ Required fields must be present: `max_model_len`, `temperature`, `top_p`, `top_k`
- ✅ `max_model_len` must be a positive integer
- ✅ `temperature` must be a float between 0.0 and 2.0
- ✅ `top_p` must be a float between 0.0 and 1.0
- ✅ `top_k` must be a positive integer

### Task Validation

- ✅ `tasks` array must exist and contain at least one task
- ✅ Each task must have required fields: `harness`, `tag`, `shots`, `reps`, `concurrency`, `max_tokens`
- ✅ `max_tokens` must be ≤ `model.max_model_len` for every task
- ✅ `shots` must be a non-negative integer
- ✅ `reps` must be a positive integer
- ✅ `concurrency` must be a positive integer

### Validation Component

Validation is performed by the `validate_config` pipeline component, which runs immediately after `validate_session_id` and before any resource-intensive operations. This ensures fast failure on invalid configurations.

## Common Task Tags (lm_eval)

Common lm-eval task identifiers:

### Mathematics
- `gsm8k_platinum_cot_llama` - Grade school math with chain-of-thought
- `math_500` - MATH dataset (500 samples)
- `aime25` - American Invitational Mathematics Examination

### Knowledge & Reasoning
- `mmlu_cot_llama` - Massive Multitask Language Understanding with CoT
- `mmlu_pro_chat` - MMLU Professional variant
- `gpqa:diamond` - Graduate-level science questions

### Instruction Following & Code
- `ifeval` - Instruction following evaluation
- `lcb:codegeneration_v6` - LiveCodeBench code generation

### Long Context
- `mrcr` - Multi-turn retrieval with long contexts

Refer to the [lm-eval documentation](https://github.com/EleutherAI/lm-evaluation-harness) for a complete list of available tasks.

## Configuration Best Practices

### Choosing `max_model_len`
- Set to the model's native context window size (check model card)
- Must accommodate: few-shot examples + prompt + generation buffer
- Larger values require more GPU memory
- Common values: `8192`, `16384`, `32768`, `131072`

### Choosing `max_tokens`
- Standard tasks (non-reasoning): `8192` - `16000`
- Reasoning tasks (with chain-of-thought): `32000` - `65000`
- Code generation tasks: `4096` - `8192`
- Always ensure: `max_tokens + 8192 (buffer) ≤ max_model_len`

### Choosing Concurrency
- Higher concurrency = higher throughput BUT more GPU memory usage
- Start with `128` for standard tasks
- Increase to `256` - `512` for short-generation tasks
- Decrease to `32` - `64` for long-generation or large models
- Monitor GPU memory usage and adjust accordingly

### Choosing Repetitions (`reps`)
- Standard evaluation: `reps=3` (provides variance estimate)
- Quick testing: `reps=1`
- High-stakes benchmarks: `reps=5` or higher
- Reasoning tasks with sampling: `reps=8` - `32` (captures solution diversity)

### Choosing Seeds
- Use consistent `base_seed` across experiments for reproducibility
- Different `base_seed` values explore different sampling trajectories
- Sequential seeds (`base_seed + rep`) ensure diverse but reproducible samples

### Tensor Parallelism (`tp`)
- Set `tp=1` for models that fit on a single GPU (~70B parameters or smaller)
- Set `tp=2` for models around 70B - 140B parameters
- Set `tp=4` or `tp=8` for larger models (200B+ parameters)
- Must be a power of 2: `1`, `2`, `4`, `8`
- The pipeline auto-adjusts if `tp` is too low based on model size

### Temperature, Top-p, Top-k
- Greedy decoding: `temperature=0.0` (deterministic)
- Standard sampling: `temperature=0.6-0.8`, `top_p=0.9`, `top_k=50`
- Creative generation: `temperature=1.0`, `top_p=0.95`, `top_k=100`
- Many benchmarks require deterministic generation - check task requirements

## Troubleshooting

### Error: "max_tokens exceeds model's max_model_len"
- **Cause**: A task's `max_tokens` is larger than `model.max_model_len`
- **Fix**: Either increase `max_model_len` or decrease `max_tokens` for the failing task

### Error: "Config file not found"
- **Cause**: Config file doesn't exist in the configs PVC at the specified path
- **Fix**: Ensure the config file is uploaded to the configs PVC and the filename matches exactly

### Error: "Missing required fields"
- **Cause**: Required fields are missing from `model` or `tasks` sections
- **Fix**: Add all required fields listed in this README

### Warning: "No lm_eval tasks found in config"
- **Cause**: No tasks have `harness: "lm_eval"` (or you're looking at a different harness component)
- **Fix**: Add at least one task with the appropriate harness identifier, or this is expected if you only have tasks for other harnesses

### vLLM Pod OOM (Out of Memory)
- **Cause**: Model + KV cache + concurrent requests exceed GPU memory
- **Fix**: 
  - Reduce `concurrency`
  - Reduce `max_model_len`
  - Increase `tp` to split model across more GPUs
  - Reduce `max_tokens`

## Storage Locations

- **Config files**: Upload to the configs PVC (`evaluation-pipeline-configs-tier-2`) in the `machine-learning` namespace
- **Example config**: `references/example_config.json` in this repository
- **Pipeline parameter**: Pass config filename (not full path) as `config_filename` parameter

## Related Documentation

- Pipeline architecture: `CLAUDE.md`
- Component details: Individual component files in `components/`
- lm-eval tasks: https://github.com/EleutherAI/lm-evaluation-harness
- vLLM parameters: https://docs.vllm.ai/
