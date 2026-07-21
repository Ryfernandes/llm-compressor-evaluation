def get_vllm_server_script():
    return '''
set -eu

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p \
  "$HOME" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$VLLM_CACHE_ROOT"

if [ "$IS_LOCAL_MODEL" = "true" ]; then
  echo "[ERROR] Local model serving is not yet implemented."
  echo "[ERROR] Please provide a HuggingFace model ID instead."
  exit 1
else
  export HF_HOME="/tier1/hf-hub"
  export HF_HUB_CACHE="/tier1/hf-hub"
  mkdir -p "$HF_HOME"
  MODEL_PATH="${MODEL}"
fi

echo "================================================================"
echo " Starting vLLM Server"
echo " Model: ${MODEL}"
echo " Model path: ${MODEL_PATH}"
echo " TP=${TP}, DP=${DP}"
echo "================================================================"

VLLM_ARGS=(
  "$MODEL_PATH"
  --served-model-name "$MODEL"
  --tensor-parallel-size "${TP}"
  --data-parallel-size "${DP}"
  --max-model-len "${MAX_MODEL_LEN}"
  --load-format safetensors
  --host 0.0.0.0
  --port 8000
)

if [ -n "${REASONING_PARSER}" ]; then
  VLLM_ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi

vllm serve "${VLLM_ARGS[@]}" > >(tee -a "${LOGS_PATH}") 2>&1 &
VLLM_PID=$!
trap 'kill -TERM $VLLM_PID 2>/dev/null; wait $VLLM_PID' SIGTERM
wait $VLLM_PID
'''
