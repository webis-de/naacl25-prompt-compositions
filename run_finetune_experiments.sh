#! /bin/bash

set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TRITON_CACHE_DIR="/tmp"  # should make deepspeed a bit faster

TIMESTAMP=$(date +"%Y%m%d%H%M%S")

BASE_MODEL_NAME="deberta-v3-large"

SEEDS_FILE_PATH="src/bias_detection/config/seeds.txt"
SEEDS=()
while IFS= read -r line || [[ -n "$line" ]]; do
    SEEDS+=("$line")
done < "${SEEDS_FILE_PATH}"

DATASETS=(
    "sbic"
    "stereoset"
    "cobra_frames"
    "semeval"
    "esnli"
    "commonsense_qa"
)


######################################################################
# TRAINING PHASE

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        {
            echo -e "\e[31m==================================================\e[0m"
            echo -e "\e[31mStarting finetuning for seed ${seed}\e[0m"
            echo -e "\e[31mDataset: ${dataset}\e[0m"
            echo -e "\e[31m==================================================\e[0m"

            MODEL_IDENTIFIER="finetune_${TIMESTAMP}_${dataset}"

            accelerate launch --config_file src/bias_detection/accelerate_finetune_config.yaml \
                src/bias_detection/finetune-train.py \
                --base_model "./models/${BASE_MODEL_NAME}" \
                --cache_path "/tmp" \
                --model_identifier "${MODEL_IDENTIFIER}" \
                --seed "${seed}" \
                --data "${dataset}"

            echo ""
            echo ""
        }
    done
done


######################################################################
# INFERENCE PHASE

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        {
            MODEL_IDENTIFIER="finetune_${TIMESTAMP}_${dataset}"

            echo -e "\e[31m==================================================\e[0m"
            echo -e "\e[31mStarting inference for seed ${seed}\e[0m"
            echo -e "\e[31mBase model: ${BASE_MODEL_NAME}\e[0m"
            echo -e "\e[31mModel identifier: ${MODEL_IDENTIFIER}\e[0m"
            echo -e "\e[31m==================================================\e[0m"

            python src/bias_detection/finetune-inference.py \
                --base_model "./models/${BASE_MODEL_NAME}" \
                --model "./models/${BASE_MODEL_NAME}-${MODEL_IDENTIFIER}-seed${seed}" \
                --cache_path "/tmp" \
                --data "${dataset}"

            echo ""
            echo ""
        }
    done
done
