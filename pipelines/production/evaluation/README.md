# Evaluation Pipeline Configuration Guide

This document describes the configuration file format for the LLM evaluation pipeline.

## Overview

The evaluation pipeline uses two separate JSON configuration files:
- **Model config**: Model serving and generation parameters (vLLM server configuration, sampling parameters)
- **Evaluation config**: Evaluation tasks to run across different harnesses

Configuration files are stored in the configs PVC under separate subdirectories:
- Model configs: `/configs/model/<model_config_name>`
- Evaluation configs: `/configs/evaluation/<evaluation_config_name>`

When launching the pipeline, provide the filenames (not full paths) for both `model_config_name` and `evaluation_config_name`.

## Model Configuration

Stored in `/configs/model/`. Contains model serving and generation parameters at the top level.

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `max_model_len` | integer | Maximum sequence length (context + generation) supported by the model. Must be >= largest `max_tokens` across all tasks. | `16384` |
| `temperature` | float | Sampling temperature for generation. Higher values = more random. Range: `0.0` to `2.0`. | `0.8` |
| `top_p` | float | Nucleus sampling parameter. Only tokens with cumulative probability <= `top_p` are considered. Range: `0.0` to `1.0`. | `0.95` |
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
  "reasoning_parser": "",
  "max_model_len": 16384,
  "tp": 1,
  "dp": 1,
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 50
}
```

## Evaluation Configuration

Stored in `/configs/evaluation/`. Contains the `tasks` array specifying evaluation tasks to run.

### Required Fields (All Harnesses)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `harness` | string | Evaluation harness identifier. Currently supported: `"lm_eval"`, `"lighteval"`. | `"lm_eval"` |
| `tag` | string | Task identifier within the harness (e.g., lm-eval task name). | `"gsm8k_platinum_cot_llama"` |
| `shots` | integer | Number of few-shot examples to include in the prompt. Range: `0` to `10`. | `5` |
| `reps` | integer | Number of evaluation repetitions with different seeds. Each rep produces independent results. | `3` |
| `concurrency` | integer | Number of concurrent inference requests. Higher values increase throughput but require more GPU memory. | `256` |
| `max_tokens` | integer | Maximum number of tokens to generate per example. Must be <= `max_model_len` in the model config. | `16000` |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_seed` | integer | `1234` | Base random seed for the first repetition. Subsequent reps use `base_seed + rep_number`. |
| `timeout` | integer | `1800` | Request timeout in seconds for each inference call. |

### Task Filtering by Harness

Each evaluation component filters tasks based on the `harness` field:
- `lm_eval_evaluation` component: processes tasks where `harness == "lm_eval"`
- `lighteval_evaluation` component: processes tasks where `harness == "lighteval"`

This allows a single evaluation config to specify tasks for multiple evaluation frameworks.

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

**Model config** (`/configs/model/qwen3-8b.json`):

```json
{
  "reasoning_parser": "",
  "max_model_len": 16384,
  "tp": 1,
  "dp": 1,
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 50
}
```

**Evaluation config** (`/configs/evaluation/standard-benchmarks.json`):

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
    }
  ]
}
```

## Validation Rules

The pipeline validates both configuration files before starting expensive operations (GPU pod creation). Validation failures will cause the pipeline to fail immediately with a descriptive error message.

### Model Config Validation

- Required fields must be present: `max_model_len`, `temperature`, `top_p`, `top_k`
- `max_model_len` must be a positive integer
- `temperature` must be a float between 0.0 and 2.0
- `top_p` must be a float between 0.0 and 1.0
- `top_k` must be a positive integer

### Evaluation Config Validation

- `tasks` array must exist and contain at least one task
- Each task must have required fields: `harness`, `tag`, `shots`, `reps`, `concurrency`, `max_tokens`
- `max_tokens` must be <= `max_model_len` from the model config for every task
- `shots` must be a non-negative integer
- `reps` must be a positive integer
- `concurrency` must be a positive integer