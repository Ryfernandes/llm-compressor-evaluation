#!/usr/bin/env bash

set -euo pipefail
# e for error, u for undefined variables, o pipefail to catch errors in pipelines

MODEL_NAME="Qwen/Qwen3-8B"
HOST="0.0.0.0"
PORT="8000"
BASE_URL="http://${HOST}:${PORT}/v1"

TASK_NAME="aime_25"
TASK_TAG="aime_25"
SHOTS=0
REPS=8
VENV="lighteval-eldar-fix-fork"
MAX_GEN_TOKS=32768

mkdir -p results
mkdir -p logs

cleanup() {
    echo "vLLM server shutting down"
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill -- "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
# Use vLLM PID if it exists, otherwise an empty string
# Kills the process and wait for it to exit, ignoring errors (for the -e)

activate_venv() {
    local env_name="$1"

    cd "../environments/${env_name}"

    if [ ! -d ".venv" ]; then
        uv venv
    fi

    source ".venv/bin/activate"
    uv sync
    cd -
}
# Activates the appropriate virtual environment

trap cleanup EXIT
# When the script exits, guarantee that cleanup is called

echo "Activating venv ${VENV}"

activate_venv "$VENV"

setsid chg run --gpus 1 -- vllm serve "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --reasoning-parser deepseek_r1 \
    > logs/${TASK_NAME}_vllm_server.log 2>&1 &
# Log everything in logs/${TASK_NAME}_vllm_server.log, both errors and normal logs

VLLM_PID=$!
# & At the end starts vLLM in the background. $! gets the PID of the last background process

echo "vLLM PID: $VLLM_PID"
echo "Waiting for vLLM server"

until curl -sf "${BASE_URL}/models" > /dev/null; do
    sleep 2
done
# Does curl (HTTP request) in silent mode and treats HTTP errors as failures

echo "Server ready"

# --load-format when it is necessary to specify safetensors, pt, gguf, bitsandbytes, or dummy (random)
# --config-format for reading config as hf or mistral explicitly
# --tokenizer-mode for controlling tokenizer implementation, only if needs to be set to slow or mistral
# --tool-call-parser to allow model to produce structured tool calls, use in conjunction with --enable-auto-tool-choice.
#   Usually not set, but will just have value of the parser name
# --reasoning-parser when using a reasoning model with hidden/structured reasoning sections, but lm_eval also tends to
#   have a think_end_token

# --tensor-parallel-size can be used for running on multiple GPUs

for i in $(seq 1 "$REPS"); do
    echo "Evaluation Run $i/$REPS"

    SEED=$((1233 + i))

    chg run --gpus 1 -- lighteval endpoint litellm \
        "model_name=hosted_vllm/${MODEL_NAME},provider=hosted_vllm,base_url=${BASE_URL},timeout=3600,concurrent_requests=8,generation_parameters={temperature:0.6,max_new_tokens:${MAX_GEN_TOKS},top_p:0.9,seed:${SEED},top_k:50}" \
        "${TASK_TAG}@k=1@n=1|0" \
        --remove-reasoning-tags \
        --output-dir results/${TASK_NAME}_${i}_seed_${SEED} \
        --save-details

done

# max_new_tokens is the maximum number of tokens the model is allowed to generate per answer
# concurrent_requests is the number of API requests that lm_eval sends at the same time
# timeout is the per-request api timeout

# --seed is for reproducibility

echo "Completed evaluation"