import argparse
import logging
import os
import time
from datetime import timedelta
from typing import Dict, List, Tuple, Union

import pandas as pd
from flex_infer import VLLM, GenerationParams, TransformersLLM
from flex_infer.evaluation import evaluate_binary_classification
from icecream import ic

from src.bias_detection.config import (
    BIAS_DETECTION_LOG_PROBS,
    COBRA_PROMPT_SETTINGS,
    FEW_SHOT_COMPONENTS,
    FILE_NAME_TEMPLATES,
    LOGGING,
    PATHS,
    RANDOM_SEED,
    RESPONSE_REGEX,
    Binary_bias_labels,
    Cobra_target_groups,
)
from src.bias_detection.data_handler import DataHandler
from src.bias_detection.utils import (
    combine_results,
    create_directories,
    generate_custom_json_regex,
    safe_get,
    save_results_to_file,
    save_results_to_json,
    save_string_to_txt,
)
from src.bias_detection.utils.experiments import (
    create_bias_prompts_from_base_template,
    create_component_combinations,
    get_system_prompt,
    load_model,
    parse_bias_detection_arguments,
    should_skip_combination,
)
from src.prompt_templates import PredictionSchema


def generate_output(
    prompts: List[str],
    model: Union[VLLM, TransformersLLM],
    seed: int,
    decoding_strategy: str,
    system_prompt: str,
) -> Tuple[List[str], List[Dict[str, float]], List[int]]:
    """Generate outputs for a single generation mode."""
    bias_choices = [label.value for label in Binary_bias_labels]

    if decoding_strategy != "greedy":
        raise ValueError(f"Invalid decoding strategy for this experiment: {decoding_strategy}")

    generation_params = GenerationParams(
        temperature=0.0,
        seed=seed,
        max_tokens=32,
        logprobs=BIAS_DETECTION_LOG_PROBS,
    )

    model_output = model.generate(
        prompts,
        generation_params,
        custom_regex=generate_custom_json_regex(PredictionSchema, RESPONSE_REGEX),
        system_prompt=system_prompt,
        use_tqdm=tqdm_flag,
        return_type="model_output",
    )

    converted_output = model.convert_output_str_to_json(model_output)

    # invalid JSON is logged and converted to a default answer
    script_logger.info(f"Invalid output: {converted_output['invalid_outputs']}")
    print(f"Invalid output: {converted_output['invalid_outputs']}")
    default_answer = {"isBiased": "no"}
    fixed_output = [
        output if not isinstance(output, str) else default_answer
        for output in converted_output["json_output"]
    ]

    # the model predictions are extracted from the JSON output and binarized for later evaluation
    extracted_output = [safe_get(obj, "isBiased") for obj in fixed_output]
    predicted_labels = [
        1 if o == Binary_bias_labels.POSITIVE.value else 0 for o in extracted_output
    ]

    answer_token_distributions = model.get_output_distribution(model_output, bias_choices)

    return extracted_output, answer_token_distributions, predicted_labels


def create_result_path(result_type: str, prefix: str, model_name: str, combinations: str) -> str:
    """Create the path for the result files."""

    combinations = sorted(list(combinations))
    components = "_".join(combinations)

    return FILE_NAME_TEMPLATES[result_type].format(
        prefix=prefix, model_name=model_name, components=components
    )


def save_results(
    args: argparse.Namespace,
    data: pd.DataFrame,
    output: List[str],
    label_dist: List[Dict[str, float]],
    predicted_labels: List[int],
    seed: int,
    combinations: Tuple[str],
    prompts: List[str],
) -> None:
    """Save the results of the experiment to a JSON file and the output to a CSV file."""
    pos_label, neg_label = Binary_bias_labels.POSITIVE.value, Binary_bias_labels.NEGATIVE.value
    results = evaluate_binary_classification(data["labels"], predicted_labels)
    combined_results = combine_results(seed, results, output)
    script_logger.info(f"Evaluation results: {results}")

    df = pd.DataFrame(
        {
            "input": data["posts"],
            "output": predicted_labels,
            "output_prob_dist": [{"1": p[pos_label], "0": p[neg_label]} for p in label_dist],
            "true_label": data["labels"],
        }
    )

    # create file paths and save results
    output_path = create_result_path("outputs", args.prefix, args.model_name, combinations)
    json_path = create_result_path("results", args.prefix, args.model_name, combinations)
    txt_path = create_result_path("prompt_template", args.prefix, args.model_name, combinations)

    save_results_to_json(combined_results, json_path)
    save_results_to_file(df, output_path, seed, file_type="parquet")
    save_string_to_txt(prompts[0], txt_path)


def main(args: argparse.Namespace):
    split_map = {"train": "train", "test": "test", "dev": "dev", "val": "dev"}
    split = split_map[args.split]

    script_logger.info(">>> Prepare combinations of prompt components...")
    prompt_component_combinations = create_component_combinations(args)
    script_logger.info(f"Number of combinations: {len(prompt_component_combinations)}")
    script_logger.info(f"Prompt Component combinations: {prompt_component_combinations}")

    print(">>> Loading Cobra Frames data ...")
    data_handler = DataHandler(datasets_to_load=["cobra_frames"])
    cobra_data = data_handler.load_cobra_frames_data_as_dict()[split]
    script_logger.info(f"Cobra Frames loaded: {cobra_data.keys()}")

    print(f">>> Loading {args.model_name}...")
    llm = load_model(args, random_seed=RANDOM_SEED[0])

    for combination in prompt_component_combinations:
        # create paths for the result files to check if the experiment already exists
        result_path = create_result_path("results", args.prefix, args.model_name, combination)

        # check if this combination was already processed
        should_skip, seeds_to_skip = should_skip_combination(result_path, combination)
        if should_skip:
            continue

        for seed in args.seeds:
            # check if this specific seed was already processed
            if int(seed) in seeds_to_skip:
                print(f"Skipping seed {seed} for combination {sorted(list(combination))}...")
                continue

            script_logger.info(f"Combination: {combination}, Seed: {seed}")
            script_logger.info(">>> Create prompts and system prompt...")
            prompts = create_bias_prompts_from_base_template(
                combination,
                cobra_data,
                eos_token=llm.eos_token,
                description_type=COBRA_PROMPT_SETTINGS["description_type"],
                definition_type=COBRA_PROMPT_SETTINGS["definition_type"],
                stimulus_type=COBRA_PROMPT_SETTINGS["stimulus_type"],
                seed=seed,
                return_format_example=PredictionSchema,
                all_categories=[category.value for category in Cobra_target_groups],
                dataset_name="cobra",
            )
            system_prompt = get_system_prompt(
                combination,
                COBRA_PROMPT_SETTINGS["system_prompt_type"],
                COBRA_PROMPT_SETTINGS["system_prompt_name"],
            )
            script_logger.info(f"System Prompt: {system_prompt!r}")

            print(f">>> Generating outputs for seed {seed} and components {combination}...")
            if args.generation_mode == "single-generation":
                output, label_dist, pred_label = generate_output(
                    prompts, llm, seed, args.decoding_strategy, system_prompt
                )
            else:
                raise ValueError(f"Invalid generation mode: {args.generation_mode}")

            script_logger.info(">>> Evaluating and saving outputs...")

            save_results(
                args, cobra_data, output, label_dist, pred_label, seed, combination, prompts
            )
            script_logger.info(f"===== Outputs saved for seed {seed} and components {combination}")

            if not set(FEW_SHOT_COMPONENTS).intersection(set(combination)):
                # we only want different seeds during greedy decoding for the selection and ordering
                # of the few shot examples. The outputs for all other component combinations are
                # the same for all seeds (greedy decoding).
                break


if __name__ == "__main__":
    start_time = time.time()

    args = parse_bias_detection_arguments()

    # debug mode with icecream
    ic.enable() if os.getenv("IC_DEBUG") == "True" else ic.disable()
    tqdm_flag = False if os.getenv("TQDM_DISABLE") == "True" else True

    create_directories([PATHS["logs"], PATHS["results"], PATHS["outputs"], PATHS["intermediate"]])

    # set up logging
    logging.basicConfig(
        filename=FILE_NAME_TEMPLATES["logs"].format(
            prefix=args.prefix,
            script_name=os.path.basename(__file__).replace(".py", "").replace("_", "-"),
            model_name=args.model_name,
        ),
        level=LOGGING["level"],
        format=LOGGING["format"],
        datefmt=LOGGING["datefmt"],
        filemode=LOGGING["write_mode"],
    )

    script_logger = logging.getLogger(os.path.basename(__file__))
    script_logger.info(f"##### Run {os.path.basename(__file__)}")
    main(args)
    script_logger.info(f"Script completed in {timedelta(seconds=round(time.time() - start_time))}")
