#! /bin/bash

set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TRITON_CACHE_DIR="/tmp"  # should make deepspeed a bit faster

SEEDS_FILE_PATH="src/bias_detection/config/seeds.txt"
readarray -t SEEDS < "${SEEDS_FILE_PATH}"

TIMESTAMP=$(date +"%Y%m%d%H%M%S")

BASE_MODEL="deberta-v3-large"

##### EXPERIMENTS SETTINGS #####
training_datasets=(
    "sbic"
    "stereoset"
    "cobra_frames"
    "semeval"
    "esnli"
    "commonsense_qa"
)
inference_datasets=(
    "sbic"
    "stereoset"
    "cobra_frames"
    "semeval"
    "esnli"
    "commonsense_qa"
)

components=(
    "category-few-shot"
    "category-few-shot;definitions"
    "category-few-shot;definitions;directional-stimulus"
    "category-few-shot;definitions;directional-stimulus;system-prompts"
    "category-few-shot;definitions;system-prompts"
    "category-few-shot;directional-stimulus"
    "category-few-shot;directional-stimulus;system-prompts"
    "category-few-shot;system-prompts"
    "definitions"
    "definitions;directional-stimulus"
    "definitions;directional-stimulus;random-few-shot"
    "definitions;directional-stimulus;random-few-shot;system-prompts"
    "definitions;directional-stimulus;similar-few-shot"
    "definitions;directional-stimulus;similar-few-shot;system-prompts"
    "definitions;directional-stimulus;system-prompts"
    "definitions;random-few-shot"
    "definitions;random-few-shot;system-prompts"
    "definitions;similar-few-shot"
    "definitions;similar-few-shot;system-prompts"
    "definitions;system-prompts"
    "directional-stimulus"
    "directional-stimulus;random-few-shot"
    "directional-stimulus;random-few-shot;system-prompts"
    "directional-stimulus;similar-few-shot"
    "directional-stimulus;similar-few-shot;system-prompts"
    "directional-stimulus;system-prompts"
    "random-few-shot"
    "random-few-shot;system-prompts"
    "similar-few-shot"
    "similar-few-shot;system-prompts"
    "system-prompts"
    "task-description-only"
    "cot;category-few-shot"
    "cot;category-few-shot;definitions"
    "cot;category-few-shot;definitions;directional-stimulus"
    "cot;category-few-shot;definitions;directional-stimulus;system-prompts"
    "cot;category-few-shot;definitions;system-prompts"
    "cot;category-few-shot;directional-stimulus"
    "cot;category-few-shot;directional-stimulus;system-prompts"
    "cot;category-few-shot;system-prompts"
    "cot;definitions"
    "cot;definitions;directional-stimulus"
    "cot;definitions;directional-stimulus;random-few-shot"
    "cot;definitions;directional-stimulus;random-few-shot;system-prompts"
    "cot;definitions;directional-stimulus;similar-few-shot"
    "cot;definitions;directional-stimulus;similar-few-shot;system-prompts"
    "cot;definitions;directional-stimulus;system-prompts"
    "cot;definitions;random-few-shot"
    "cot;definitions;random-few-shot;system-prompts"
    "cot;definitions;similar-few-shot"
    "cot;definitions;similar-few-shot;system-prompts"
    "cot;definitions;system-prompts"
    "cot;directional-stimulus"
    "cot;directional-stimulus;random-few-shot"
    "cot;directional-stimulus;random-few-shot;system-prompts"
    "cot;directional-stimulus;similar-few-shot"
    "cot;directional-stimulus;similar-few-shot;system-prompts"
    "cot;directional-stimulus;system-prompts"
    "cot;random-few-shot"
    "cot;random-few-shot;system-prompts"
    "cot;similar-few-shot"
    "cot;similar-few-shot;system-prompts"
    "cot;system-prompts"
    "cot;task-description-only"
)

##### SET UP MODELS #####
declare -A models=(
    ["mistral-7b-instruct-v2"]="../models/mistral-7b-instruct-v02"
    ["command-r-v01"]="../models/command-r-v01"
    ["llama3-70b-instruct"]="../models/llama3-70b-instruct"
)

######################################################################
##### RUN TRAINING & INFERENCE #####
for model_name in "${!models[@]}"; do
    for dataset in "${training_datasets[@]}"; do
        for seed in "${SEEDS[@]}"; do
            {
                echo -e "\e[31m==================================================\e[0m"
                echo -e "\e[31mStarting composition prediction training for:\e[0m"
                echo -e "\e[31mLLM: ${model_name}\e[0m"
                echo -e "\e[31mBase model: ${BASE_MODEL}\e[0m"
                echo -e "\e[31mDataset: ${dataset}\e[0m"
                echo -e "\e[31mSeed: ${seed}\e[0m"
                echo -e "\e[31m==================================================\e[0m"

                MODEL_IDENTIFIER="composition-prediction-for-${model_name}-on-${dataset}_${TIMESTAMP}-seed${seed}"
                python src/components_prediction/transformer-train.py \
                    --model_name "$(pwd)/models/${BASE_MODEL}" \
                    --llm-name "${model_name}" \
                    --prefix "" \
                    --seed "${seed}" \
                    --model_identifier "${MODEL_IDENTIFIER}" \
                    --prompt_components "${components[@]}" \
                    --data "${dataset}"

                echo ""
                echo ""
            }
        done
    done
done

######################################################################
# INFERENCE PHASE
for model_name in "${!models[@]}"; do
    for training_dataset in "${training_datasets[@]}"; do
        for inference_dataset in "${inference_datasets[@]}"; do
            for seed in "${SEEDS[@]}"; do
                {
                    echo -e "\e[31m==================================================\e[0m"
                    echo -e "\e[31mStarting composition prediction inference for:\e[0m"
                    echo -e "\e[31mLLM: ${model_name}\e[0m"
                    echo -e "\e[31mBase model: ${BASE_MODEL}\e[0m"
                    echo -e "\e[31mTraining data: ${training_dataset}\e[0m"
                    echo -e "\e[31mInference data: ${inference_dataset}\e[0m"
                    echo -e "\e[31mSeed: ${seed}\e[0m"
                    echo -e "\e[31m==================================================\e[0m"

                    MODEL_IDENTIFIER="composition-prediction-for-${model_name}-on-${training_dataset}_${TIMESTAMP}-seed${seed}"
                    python src/components_prediction/transformer-inference.py \
                        --model_name "$(pwd)/models/${BASE_MODEL}" \
                        --seed "${seed}" \
                        --model_identifier "${MODEL_IDENTIFIER}" \
                        --data "${inference_dataset}"

                    echo ""
                    echo ""
                }
            done
        done
    done
done
