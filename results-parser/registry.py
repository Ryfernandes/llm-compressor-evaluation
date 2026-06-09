METRICS_REGISTRY = {
    "gsm8k_platinum_cot_llama": [
        "exact_match,strict-match",
    ],
    "mmlu_cot_llama": [
        "exact_match,strict_match",
    ],
    "mmlu_pro_chat": [
        "exact_match,custom-extract",
    ],
    "ifeval": [
      "inst_level_strict_acc,none"
    ],
    "math_500|0": [
        "pass@k:k=1&n=1",
    ],
    "aime25|0": [
        "pass@k:k=1&n=1",
    ],
    "gpqa:diamond|0": [
        "gpqa_pass@k:k=1&n=1",
    ],
    "lcb:codegeneration_v6": [
        "codegen_pass@k:k=1&n=1",
    ],
    "mrcr": [
        "score_gt_16k_le_32k",
        "AUC"
    ]
}