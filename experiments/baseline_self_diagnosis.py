import argparse
import logging
import os
import time
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd
from flex_infer import VLLM, GenerationParams
from flex_infer.evaluation import evaluate_binary_classification
from icecream import ic

from src.bias_detection.config import (
    BIAS_DETECTION_LOG_PROBS,
    FILE_NAME_TEMPLATES,
    LOGGING,
    MAX_LOG_PROBS,
    PATHS,
    RANDOM_SEED,
)
from src.bias_detection.data_handler import DataHandler
from src.bias_detection.utils import (
    create_directories,
    save_results_to_file,
    save_results_to_json,
    save_string_to_txt,
)
from src.prompt_templates import baseline_prompt, baseline_prompt_semeval


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the experiment.

    Returns:
        argparse.Namespace: Parsed arguments with dataset, split, and model_path.
    """
    parser = argparse.ArgumentParser(description="Parse command line arguments for the experiment.")

    parser.add_argument("--dataset", type=str, required=True, help="The name of the dataset to use")

    parser.add_argument(
        "--split", type=str, required=True, help="The data split to use for the experiment"
    )

    parser.add_argument(
        "--model_path", type=str, required=True, help="The path to the gpt2-xl model"
    )

    args = parser.parse_args()

    valid_splits = ["train", "test", "dev"]
    if args.split not in valid_splits:
        parser.error(f"Invalid split argument: {args.split}. Must be one of {valid_splits}.")

    valid_datasets = ["sbic", "stereoset", "cobra_frames", "semeval"]
    if args.dataset not in valid_datasets:
        parser.error(f"Invalid dataset argument: {args.dataset}. Must be one of {valid_datasets}.")

    if args.dataset == "sbic":
        split_mapping = {"train": "train_sub_split_balanced"}

        if args.split in split_mapping:
            args.split = split_mapping[args.split]

    return args


def load_model(model_path: str) -> VLLM:
    """
    Load the GPT2-XL model from the specified path.

    Args:
        model_path (str): Path to the GPT2-XL model.

    Returns:
        VLLM: Loaded model.
    """
    model_settings = {
        "name": "gpt2-xl",
        "model_path": model_path,
        "prompt_format": "no_style",
        "seed": RANDOM_SEED[0],
        "quant": None,
        "num_gpus": 1,
        "max_logprobs": MAX_LOG_PROBS,
    }

    return VLLM(**model_settings)


def load_dataset(dataset_name: str, split: str) -> Dict[str, List[Any]]:
    """
    Load the dataset based on the name and split.

    Args:
        dataset_name (str): Name of the dataset to load.
        split (str): Data split to use (train, test, dev).

    Returns:
        Dict[str, List[Any]]: Loaded dataset.
    """
    data_handler = DataHandler([dataset_name])

    if dataset_name == "sbic":
        return data_handler.load_sbic_data_as_dict()[split]
    elif dataset_name == "stereoset":
        return data_handler.load_stereoset_data_as_dict()[split]
    elif dataset_name == "cobra_frames":
        return data_handler.load_cobra_frames_data_as_dict()[split]
    elif dataset_name == "semeval":
        return data_handler.load_semeval_data_as_dict()[split]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")


def build_prompts(data: Dict[str, List[Any]], dataset_name: str) -> List[str]:
    """
    Build prompts for the model based on the dataset. The prompts use the same template as described
    in the self-diagnosis paper.

    Args:
        data (Dict[str, List[Any]]): The dataset.
        dataset_name (str): The name of the dataset.

    Returns:
        List[str]: List of prompts.
    """
    if dataset_name == "semeval":
        return [baseline_prompt_semeval(input=p) for p in data["posts"]]

    if dataset_name in ["sbic", "cobra_frames"]:
        detection_type = "social bias"
    else:
        detection_type = "stereotypes"

    return [baseline_prompt(input=p, detection_type=detection_type) for p in data["posts"]]


def generate_output(
    model: VLLM, prompts: List[str], dataset_name: str
) -> Tuple[List[str], List[Dict[str, float]], List[int]]:
    """
    Generate output from the model based on the prompts.

    Args:
        model (VLLM): The loaded model.
        prompts (List[str]): List of prompts.

    Returns:
        Tuple[List[str], List[Dict[str, float]], List[int]]: Generated outputs, token distributions,
        and predicted labels.
    """
    if dataset_name == "semeval":
        answer_choices = ["positive", "negative"]
    else:
        answer_choices = ["yes", "no"]

    generation_params = GenerationParams(
        temperature=0.0,
        seed=RANDOM_SEED[0],
        max_tokens=16,
        logprobs=BIAS_DETECTION_LOG_PROBS,
    )

    model_output = model.generate(
        prompts,
        generation_params,
        choices=answer_choices,
        use_tqdm=tqdm_flag,
        return_type="model_output",
    )

    output = [o.lower() for o in model_output.output]
    if dataset_name == "semeval":
        predicted_labels = [1 if o == "positive" else 0 for o in output]
    else:
        predicted_labels = [1 if o == "yes" else 0 for o in output]

    answer_token_distributions = model.get_output_distribution(model_output, answer_choices)

    return output, answer_token_distributions, predicted_labels


def evaluate_model_output(
    data: Dict[str, List[Any]], predicted_labels: List[int], output: List[str], dataset_name: str
) -> Dict[str, Any]:
    """
    Evaluate the model output against the true labels.

    Args:
        data (Dict[str, List[Any]]): The dataset containing true labels.
        predicted_labels (List[int]): The predicted labels from the model.
        output (List[str]): The output strings from the model.

    Returns:
        Dict[str, Any]: Evaluation results and output statistics.
    """
    classification_results = evaluate_binary_classification(data["labels"], predicted_labels)

    if dataset_name == "semeval":
        output_distribution = {
            "positive": output.count("positive"),
            "negative": output.count("negative"),
        }
    else:
        output_distribution = {
            "no": output.count("no"),
            "yes": output.count("yes"),
        }

    output_stats = {
        "num_outputs": len(output),
        "output_set": sorted(list(set(output))),
        "output_distribution": output_distribution,
    }

    return {"eval_results": classification_results, "output_stats": output_stats}


def save_results(
    data: Dict[str, List[Any]],
    predicted_labels: List[int],
    answer_token_dist: List[Dict[str, float]],
    eval_results: Dict[str, float],
    dataset: str,
    split: str,
    prompts: List[str],
) -> None:
    """
    Save the results to files.

    Args:
        data (Dict[str, List[Any]]): The dataset.
        predicted_labels (List[int]): The predicted labels.
        answer_token_dist (List[Dict[str, float]]): The token distributions for each answer.
        eval_results (Dict[str, float]): The evaluation results.
        dataset (str): The name of the dataset.
        split (str): The data split used.
        prompts (List[str]): The list of prompts used.
    """
    split_mapping = {"train_sub_split_balanced": "train"}
    if split in split_mapping:
        split = split_mapping[split]

    output_path = f"outputs/baseline_self_diagnosis_{dataset}_{split}.parquet"
    result_path = f"results/baseline_self_diagnosis_{dataset}_{split}.json"
    prompt_path = f"tmp/baseline_self_diagnosis_{dataset}_{split}.txt"

    if dataset == "semeval":
        df = pd.DataFrame(
            {
                "input": data["posts"],
                "output": predicted_labels,
                "output_prob_dist": [
                    {"1": p["positive"], "0": p["negative"]} for p in answer_token_dist
                ],
                "true_label": data["labels"],
            }
        )
    else:
        df = pd.DataFrame(
            {
                "input": data["posts"],
                "output": predicted_labels,
                "output_prob_dist": [{"1": p["yes"], "0": p["no"]} for p in answer_token_dist],
                "true_label": data["labels"],
            }
        )

    results = {RANDOM_SEED[0]: eval_results}
    save_results_to_json(results, result_path)
    save_results_to_file(df, output_path, RANDOM_SEED[0], file_type="parquet")
    save_string_to_txt(prompts[0], prompt_path)

    return None


def main(args: argparse.Namespace) -> None:
    print(f">>> Loading '{args.dataset}' with '{args.split}' split ...")
    data = load_dataset(args.dataset, args.split)

    print(f">>> Loading GPT2-XL from '{args.model_path}' ...")
    gpt2 = load_model(args.model_path)

    print(f">>> Building {len(data['posts'])} prompts ...")
    prompts = build_prompts(data, args.dataset)

    print(f"Generating output for {len(prompts)} prompts ...")
    output, answer_token_distributions, predicted_labels = generate_output(
        gpt2, prompts, args.dataset
    )

    print(">>> Evaluating model predictions ...")
    evaluation_results = evaluate_model_output(data, predicted_labels, output, args.dataset)

    save_results(
        data=data,
        predicted_labels=predicted_labels,
        answer_token_dist=answer_token_distributions,
        eval_results=evaluation_results,
        dataset=args.dataset,
        split=args.split,
        prompts=prompts,
    )


if __name__ == "__main__":
    start_time = time.time()

    # debug mode with icecream
    ic.enable() if os.getenv("IC_DEBUG") == "True" else ic.disable()
    tqdm_flag = False if os.getenv("TQDM_DISABLE") == "True" else True

    create_directories([PATHS["logs"], PATHS["results"], PATHS["outputs"], PATHS["intermediate"]])

    args = parse_arguments()

    # set up logging
    logging.basicConfig(
        filename=FILE_NAME_TEMPLATES["logs"].format(
            prefix="",
            script_name=os.path.basename(__file__).replace(".py", "").replace("_", "-"),
            model_name="gpt2-xl",
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
