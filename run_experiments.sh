#!/bin/bash
set -e

# set and create the outlines cache directory
UNIQUE_ID=$(date +%Y%m%d%H%M%S)_$RANDOM
CACHE_DIR="$HOME/.cache/outlines_custom_cache/job_$UNIQUE_ID"
export OUTLINES_CACHE_DIR="$CACHE_DIR"
echo "Using cache directory: $CACHE_DIR"
mkdir -p "$CACHE_DIR"

start_time=$(date +%s)

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LOG_LEVEL="WARNING" # used for vLLM logging
export IC_DEBUG="False"  # icecream print debugging
export TOKENIZERS_PARALLELISM="false"  # for sentence-transformers


##### SET UP MODELS #####
declare -A models=(
    ["mistral-7b-instruct-v2"]="../models/mistral-7b-instruct-v02" # < 80 GB VRAM
    ["llama3-70b-instruct"]="../models/llama3-70b-instruct" # < 160 GB VRAM
    ["command-r-v01"]="../models/command-r-v01" # < 160 GB VRAM
)

# do not change
declare -A prompt_templates=(
    ["mistral-7b-instruct-v2"]="llama2"
    ["llama3-70b-instruct"]="llama3"
    ["command-r-v01"]="cohere"
)

# do not change
declare -A quantizations=(
    ["mistral-7b-instruct-v2"]="None"
    ["llama3-70b-instruct"]="None"
    ["command-r-v01"]="None"
)


##### DEFAULT VALUES #####
default_num_gpus=1 # number of GPUs to use (1, 2, 4)
default_experiment_script="experiments/sbic_greedy.py"
default_seeds="23,42,271,314,1337" # seeds are only used for the few-shot sampling methods
default_engine="vllm" # vllm (recommended), transformers (unstable)
default_generation_mode="single-generation" # single-generation is the only supported generation mode
default_prefix="" # optional prefix for output files to distinguish between different runs
default_decoding_strategy="greedy" # greedy is the only supported decoding strategy
default_split="train" # train, test, dev
default_gpu_mem="80gb" # do not change this even if you have more or less VRAM
default_components=("definitions" "system-prompts" "directional-stimulus")
default_sampling_methods=("similar-few-shot" "random-few-shot" "category-few-shot")


##### COMMAND LINE ARGUMENTS #####
experiment_script="${1:-$default_experiment_script}"
prefix="${2:-$default_prefix}"
split="${3:-$default_split}"
num_gpus="${4:-$default_num_gpus}"
seeds_str="${5:-$default_seeds}"
decoding_strategy="${6:-$default_decoding_strategy}"
generation_mode="${7:-$default_generation_mode}"
engine="${8:-$default_engine}"
gpu_mem="${9:-$default_gpu_mem}"
IFS=',' read -r -a seeds <<< "$seeds_str"


##### RUN EXPERIMENTS #####
for model_name in "${!models[@]}"; do
    echo "Running experiment for '$model_name' with quantization: '$quant', and GPUs: $num_gpus"
    echo "Prefix for outputs is '$prefix' (Defaults to '')"

    model_path=${models[$model_name]}
    prompt_template=${prompt_templates[$model_name]}
    quant=${quantizations[$model_name]}

    python $experiment_script \
        --prompt_components "${default_components[@]}" \
        --few_shot_sampling_methods "${default_sampling_methods[@]}" \
        --seeds "${seeds[@]}" \
        --model_name "$model_name" \
        --model_path "$model_path" \
        --prompt_template "$prompt_template" \
        --quant "$quant" \
        --num_gpus $num_gpus \
        --prefix "$prefix" \
        --engine "$engine" \
        --decoding_strategy "$decoding_strategy" \
        --generation_mode "$generation_mode" \
        --split "$split" \
        --gpu_mem "$gpu_mem"
done


end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
hours=$((elapsed_time / 3600))
minutes=$(( (elapsed_time % 3600) / 60 ))
seconds=$((elapsed_time % 60))
printf "All experiments completed. Total runtime: %02d:%02d:%02d\n" $hours $minutes $seconds

# clean up the custom outlines cache directory
if [ -d "$CACHE_DIR" ]; then
    echo "Cleaning up cache directory: $CACHE_DIR"
    rm -rf "$CACHE_DIR"
else
    echo "Cache directory not found: $CACHE_DIR"
fi
