# Make the for loop logic go in python
# Source save path, reps, model path, max length, base url, n_shots, and max gen toks from pipeline function inputs
# Have an outer loop that iterates through a lists of tasks to populate task_tag

cd $SAVE_PATH
mkdir tmp

for i in $(seq 1 "$REPS"); do
    echo "Evaluation Run $i/$REPS"

    SEED=$((1233 + i))

    lm_eval \
        --model local-chat-completions \
        --tasks ${TASK_TAG} \
        --model_args "model=${MODEL_PATH},max_length=${MAX_LENGTH},base_url=${BASE_URL}/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
        --num_fewshot $N_SHOTS \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --output_path tmp \
        --seed "$SEED" \
        --gen_kwargs "do_sample=True,temperature=0.6,top_p=0.9,top_k=50,max_gen_toks=${MAX_GEN_TOKS},seed=$SEED"
    
    json_file="$(find tmp -type f -name "*.json" | head -n 1)"

    mv "$json_file" "results/${TASK_TAG}_seed_${SEED}.json"
    rm -rf tmp/*
done

rmdir tmp