#!/usr/bin/env bash

set -euo pipefail
# e for error, u for undefined variables, o pipefail to catch errors in pipelines

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
HOST="0.0.0.0"
PORT="8000"
BASE_URL="http://${HOST}:${PORT}/v1"

TASK_NAME="ifeval"
TASK_TAG="ifeval"
SHOTS=0
REPS=3
VENV="lm-eval-base"

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
    source "../environments/${env_name}/.venv/bin/activate"
    uv sync --project "../environments/${env_name}"
}
# Activates the appropriate virtual environment

trap cleanup EXIT
# When the script exits, guarantee that cleanup is called

echo "Activating venv ${VENV}"

activate_venv "$VENV"

setsid chg run --gpus 1 -- vllm serve "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
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

    chg run --gpus 1 -- lm_eval --model local-chat-completions \
        --tasks ${TASK_TAG} \
        --model_args "model=$MODEL_NAME,max_length=8192,base_url=${BASE_URL}/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
        --num_fewshot $SHOTS \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --output_path results/${TASK_NAME}_${i}_seed_${SEED}.json \
        --seed "$SEED" \
        --gen_kwargs "do_sample=True,temperature=0.6,top_p=0.9,top_k=50,max_gen_toks=8192,seed=$SEED"
done

# max_length is the maximum length of input and output assumed by lm-eval for the API model. Separate from
#   vllm's max-model-len, which is the server-side context limit. This should be aligned with max-model-length
# num_concurrent is the number of API requests that lm_eval sends at the same time. Defaults as 1, but can
#   be increased if there is spare bandwidth
# num_retries is the number of times to retry a failed API request
# tokenized_requests is whether to send text or token ids in requests. For OpenAI compatible backends, shoudl be False
# tokenizer_backend is the tokenizer to use if it needs to tokenize requests
# timeout is the per-request api timeout

# --apply_chat_template has lm_eval format task prompts with the model's chat template
# --fewshot_as_multiturn will assign "User" and "Assistant" roles to each Q/A in the fewshot examples
# --seed is for reproducibility

# --limit only runs a subset of samples for each task

# max_gen_toks is an lm-eval parameter, which is the maximum number of tokens generated for
#   completion/generation tasks

echo "Completed evaluation"