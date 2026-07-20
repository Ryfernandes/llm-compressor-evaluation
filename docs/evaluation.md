# LLM Compressor Evaluation

[LLM Compressor](https://github.com/vllm-project/llm-compressor) is an open source library for applying researched model compression techniques (quantization, pruning, distillation, etc.) that produce smaller, faster models with minimal loss in accuracy or response quality.

The LLM Compressor team also contributes to the [Inference Optimization](https://huggingface.co/inference-optimization) page on HuggingFace, publishing compressed models including Day 0 releases and customer-specific models. Running comprehensive evaluations across these models and compression algorithms is critical for validating quality and comparing techniques.

This repository provides a set of configurable [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/) designed to run on [Red Hat OpenShift AI](https://www.redhat.com/en/products/ai/openshift-ai), replacing ad-hoc evaluation scripts with reproducible, configuration-driven workflows.

## Pipelines

All pipelines live under `pipelines/` and are compiled to versioned YAML artifacts via their respective `generator.py` scripts. Each pipeline is built with [KFP v2](https://www.kubeflow.org/docs/components/pipelines/) and `kfp-kubernetes` for Kubernetes-native operations (PVC mounts, secrets, node selectors, GPU requests).

### Evaluation

![Evaluation pipeline DAG](content/readme/evaluation.png)

Evaluates a HuggingFace or local model using [vLLM](https://github.com/vllm-project/vllm) serving with the [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) and [lighteval](https://github.com/huggingface/lighteval) harnesses.

The pipeline:
1. Validates the session, model, and evaluation configs
2. Deploys a vLLM server pod with the model and a FastAPI proxy for per-request logging
3. Saves vLLM startup statistics (model size, load time, KV cache info)
4. Runs configured evaluation tasks across both harnesses with multiple seed repetitions
5. Collects vLLM Prometheus metrics, proxy statistics, and KV cache utilization
6. Collates all results into a single `collated_results.json` with aggregate statistics
7. Opens a PR with results to a GitHub results repository
8. Tears down the vLLM server and proxy via an exit handler

Evaluation is configured via two JSON files: a **model config** (serving parameters like `max_model_len`, `temperature`, `tp`, `dp`) and an **evaluation config** (list of tasks specifying harness, shots, repetitions, concurrency, and max tokens). See the [configuration guide](pipelines/evaluation/README.md) for details.

### Oneshot Compression

![Oneshot compression pipeline DAG](content/readme/oneshot.png)

Compresses a HuggingFace model using LLM Compressor's `oneshot` API with a YAML compression recipe and calibration dataset.

The pipeline:
1. Validates the YAML recipe and generates a session ID
2. Estimates model size and validates it fits on an H100 GPU
3. Downloads and tokenizes a calibration dataset from HuggingFace
4. Runs `oneshot()` compression with the recipe and calibration data on an H100 node
5. Performs a smoke test generation and saves the compressed model to a PVC
6. Cleans up the calibration dataset

### Model-Free PTQ Compression

![Model-free PTQ compression pipeline DAG](content/readme/model_free_ptq.png)

Compresses a model using LLM Compressor's `model_free_ptq` API with a quantization scheme preset. Unlike the oneshot pipeline, this requires no calibration data.

The pipeline:
1. Validates the quantization scheme name
2. Estimates model size and validates GPU fit
3. Runs `model_free_ptq()` compression with the scheme and optional layer ignore list on an H100 node
4. Saves the compressed model to a PVC

### HuggingFace Upload

![HuggingFace upload pipeline DAG](content/readme/huggingface_upload.png)

Uploads a compressed model from the models PVC to HuggingFace, with an option to delete the local copy after upload.

## Infrastructure

- **Compute:** Currently H100 GPU nodes on OpenShift AI (hopes to expand to B200 nodes if they become available)
- **Storage:** PVC-based architecture with persistent storage for artifacts, configs, and models
- **Model serving:** vLLM with a FastAPI proxy for request-level statistics logging
- **Model sources:** HuggingFace Hub and local checkpoints
- **Results:** Uploaded as PRs to a [dedicated GitHub results repository](https://github.com/Ryfernandes/llm-evaluation-pipeline-results)
