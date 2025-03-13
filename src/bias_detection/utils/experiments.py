import argparse
import itertools
import json
import os
import random
from typing import Any, Dict, List, Tuple, Union

from flex_infer import VLLM, ModelOutput, TransformersLLM
from pydantic import BaseModel

from src.bias_detection.config import (
    FEW_SHOT_COMPONENTS,
    MAX_LOG_PROBS,
    NONE_CATEGORY_NAME,
    NUM_OF_FEW_SHOT_EXAMPLES,
    PROMPT_COMPONENTS_PATHS,
    RANDOM_SEED,
    VALID_COT_MODE,
    VALID_DECODING_STRATEGIES,
    VALID_ENGINES,
    VALID_GENERATION_MODES,
    VALID_PROMPT_COMPONENTS,
    Binary_bias_labels,
    SBIC_bias_categories,
)
from src.bias_detection.utils import (
    read_json,
    sample_from_each_category,
    shuffle_list_with_seed,
    sort_categories_by_index,
)
from src.prompt_templates import (
    absa_prompt,
    base_prompt,
    cobra_chain_of_thought_prompt_prediction,
    cobra_chain_of_thought_prompt_reasoning,
    common_base_prompt,
    common_chain_of_thought_prompt_prediction,
    common_chain_of_thought_prompt_reasoning,
    esnli_base_prompt,
    esnli_chain_of_thought_prompt_prediction,
    esnli_chain_of_thought_prompt_reasoning,
    sbic_chain_of_thought_prompt_prediction,
    sbic_chain_of_thought_prompt_reasoning,
    semeval_chain_of_thought_prompt_prediction,
    semeval_chain_of_thought_prompt_reasoning,
    stereoset_chain_of_thought_prompt_prediction,
    stereoset_chain_of_thought_prompt_reasoning,
)


def parse_bias_detection_arguments() -> argparse.Namespace:
    """Parses the command-line arguments for the SBIC bias detection script."""
    parser = argparse.ArgumentParser(description="Run bias detection model inference.")
    parser.add_argument(
        "--prompt_components",
        type=str,
        nargs="+",
        help="List of prompt components to include.",
        required=False,
    )
    parser.add_argument(
        "--few_shot_sampling_methods",
        type=str,
        nargs="+",
        help="List of few shot sampling methods.",
        required=False,
    )
    parser.add_argument(
        "--chain_of_thought",
        type=str,
        default=None,
        help="Mode for Chain of Thought generation (few-shot or zero-shot).",
    )
    parser.add_argument(
        "--generation_mode",
        type=str,
        default="single-generation",
        help="Generation mode for the model (self-consistency or single-generation).",
    )
    parser.add_argument(
        "--decoding_strategy",
        type=str,
        default="greedy",
        help="Decoding strategy for the model (greedy, beam-search, top-p or top-k).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="List of seeds to use for model inference.",
        required=False,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistral-7b-instruct-v2",
        help="Human friendly name of the model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../models/mistral-7b-instruct-v02",
        help="Path to the model.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="llama2",
        help="Prompt template for the model.",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default=None,
        help="Quantization setting for the model.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for model inference.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="vllm",
        help="Engine to use for model inference (vLLM or transformers).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to add to the result and log file name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Data split to use for the experiment.",
    )
    parser.add_argument(
        "--gpu_mem",
        type=str,
        default="80gb",
        help="Amount of GPU memory to use for the model. (80gb or 20gb)",
    )

    args = parser.parse_args()

    args.engine = args.engine.lower()
    args.generation_mode = args.generation_mode.lower()
    args.chain_of_thought = (
        args.chain_of_thought.lower() if args.chain_of_thought else None
    )
    args.decoding_strategy = args.decoding_strategy.lower()

    if args.quant == "None":
        args.quant = None

    if not args.seeds:
        args.seeds = RANDOM_SEED

    # just formatting
    if args.prefix:
        args.prefix = args.prefix.replace("_", "-")
        if args.prefix[-1] != "-":
            args.prefix += "_"
        else:
            args.prefix = args.prefix[:-1] + "_"

        if args.split:
            args.prefix = args.prefix.replace("_", "-") + f"{args.split}_"

    check_bias_detection_arguments(args)
    return args


def check_bias_detection_arguments(args: argparse.Namespace) -> None:
    """
    Validates the arguments passed to the SBIC bias detection script.

    Args:
        args (argparse.Namespace): The arguments provided to the function.

    Raises:
        ValueError: Raised if an invalid prompt component is detected.
        ValueError: Raised if an unrecognized engine name is provided.
        ValueError: Raised if the generation mode is not one of the expected options.
        ValueError: Raised if the chain of thought mode is not one of the expected options.
        ValueError: Raised if an invalid few-shot sampling method is detected.
        ValueError: Raised if the decoding strategy is not one of the expected options.
        ValueError: Raised if the split is not one of the expected options.
        ValueError: Raised if the GPU memory is not one of the expected options.
    """
    for component in args.prompt_components:
        if component not in VALID_PROMPT_COMPONENTS:
            raise ValueError(
                f"Prompt component: {component} | valid: {VALID_PROMPT_COMPONENTS}"
            )

    if args.few_shot_sampling_methods:
        for sampling_method in args.few_shot_sampling_methods:
            if sampling_method not in FEW_SHOT_COMPONENTS:
                raise ValueError(
                    f"Few-shot sampling method: {sampling_method} | valid: {FEW_SHOT_COMPONENTS}"
                )

    if args.engine not in VALID_ENGINES:
        raise ValueError(f"Engine: {args.engine} | valid: {VALID_ENGINES}")

    if args.generation_mode not in VALID_GENERATION_MODES:
        raise ValueError(
            f"Generation mode: {args.generation_mode} | valid: {VALID_GENERATION_MODES}"
        )

    if args.chain_of_thought and args.chain_of_thought not in VALID_COT_MODE:
        raise ValueError(f"CoT mode: {args.chain_of_thought} | valid: {VALID_COT_MODE}")

    if args.decoding_strategy not in VALID_DECODING_STRATEGIES:
        raise ValueError(
            f"Decoding strategy: {args.decoding_strategy} | valid: {VALID_DECODING_STRATEGIES}"
        )

    if args.split not in ["train", "test", "dev", "val"]:
        raise ValueError(
            f"Split: {args.split} | valid: ['train', 'test', 'dev', 'val']"
        )

    if args.gpu_mem not in ["80gb", "20gb"]:
        raise ValueError(f"GPU memory: {args.gpu_mem} | valid: ['80gb', '20gb']")


def load_model(
    args: argparse.Namespace, random_seed: int, gpu_mem: str = "80gb"
) -> Union[VLLM, TransformersLLM]:
    """
    Load the specified LLM with the given settings. The model can be either a VLLM or a
    transformers based model.

    Args:
        args (argparse.Namespace): The command line arguments containing the model settings.
        random_seed (int): The seed used for the model.

    Returns:
        Union[VLLM, TransformersLLM]: The loaded LLM model.
    """
    model_settings = {
        "name": args.model_name,
        "model_path": args.model_path,
        "prompt_format": args.prompt_template,
        "seed": random_seed,
        "quant": args.quant,
        "num_gpus": args.num_gpus,
        "max_logprobs": MAX_LOG_PROBS,
    }

    # these settings are necessary to run unquantized ~7b models on 20gb GPUs
    if gpu_mem == "20gb":
        model_settings["gpu_memory_utilization"] = 0.99  # increased from 0.95
        model_settings["max_model_len"] = 7000  # decreased from 32k
        # model_settings["max_num_seqs"] = 16  # drastically decreased from 256
        model_settings["enforce_eager"] = (
            True  # instead of using CUDA graphs (saves memory)
        )

    if args.engine == "vllm":
        return VLLM(**model_settings)
    else:
        return TransformersLLM(**model_settings)


def create_example(
    data: Dict[str, Any], hash_value: str, label: str, dataset_name: str
) -> Dict[str, str]:
    """
    Creates a single example dictionary from the provided data.

    Args:
        data (Dict[str, Any]): The dataset containing all necessary lookup tables and properties.
        hash_value (str): The hash value used to look up specific data entries.
        label (str): The label indicating the bias category (e.g., positive or negative).
        dataset_name (str): Name of the dataset.

    Raises:
        NotImplementedError: If the dataset name is not recognized.

    Returns:
        Dict[str, str]: A dictionary representing a single example with input data and associated
        properties.
    """

    if dataset_name == "sbic":
        properties = data["property_lookup"][hash_value]
        return {
            "input": data["hash_lookup"][hash_value],
            "offensive": "Yes" if properties["offensive"] >= 0.5 else "No",
            "group": properties["group"][0] if len(properties["group"]) > 0 else "",
            "implied_statement": (
                properties["implied_statement"][0]
                if len(properties["implied_statement"]) > 0
                else ""
            ),
            "output": label,
        }
    elif dataset_name == "stereoset":
        return {
            "input": data["hash_lookup"][hash_value],
            "target": data["property_lookup"][hash_value],
            "output": label,
        }
    elif dataset_name == "cobra":
        properties = data["property_lookup"][hash_value]
        return {
            "input": data["hash_lookup"][hash_value],
            "intent": properties["intent"],
            "target_group": properties["target_group"],
            "implication": properties["implication"],
            "output": label,
        }
    elif dataset_name == "semeval":
        properties = data["property_lookup"][hash_value]
        return {
            "input": data["hash_lookup"][hash_value],
            "category": properties["category"],
            "aspect_terms": properties["aspect_terms"],
            "polarity": properties["polarities"],
            "output": label,
        }
    elif dataset_name == "common_qa":
        properties = data["property_lookup"][hash_value]
        return {
            "input": data["hash_lookup"][hash_value],
            "category": properties["category"],
            "output": label,
            "concept": properties["concept"],
            "distractor": properties["distractor"],
        }
    elif dataset_name == "esnli":
        properties = data["property_lookup"][hash_value]
        return {
            "input": data["hash_lookup"][hash_value],
            "category": properties["category"],
            "output": label,
            "explanation": properties["explanation"],
        }
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")


def create_examples(
    data: Dict[str, Any], indices: List[int], label: str, dataset_name: str
) -> List[Dict[str, str]]:
    """
    Creates a list of example dictionaries from the provided data based on specified indices.

    Args:
        data (Dict[str, Any]): The dataset containing all necessary data entries.
        indices (List[int]): A list of indices used to access specific data entries.
        label (str): The label indicating the bias category (e.g., positive or negative).
        dataset_name (str): Name of the dataset.

    Raises:
        NotImplementedError: If the dataset name is not recognized.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each representing an example with input data
        and associated properties.
    """
    if dataset_name == "sbic":
        return [
            {
                "input": data["posts"][i],
                "offensive": "Yes" if data["offensive"][i] >= 0.5 else "No",
                "group": data["group"][i][0] if len(data["group"][i]) > 0 else "",
                "implied_statement": (
                    data["implied_statement"][i][0]
                    if len(data["implied_statement"][i]) > 0
                    else ""
                ),
                "output": label,
            }
            for i in indices
        ]
    elif dataset_name == "stereoset":
        return [
            {
                "input": data["posts"][i],
                "target": data["targets"][i],
                "output": label,
            }
            for i in indices
        ]
    elif dataset_name == "cobra":
        return [
            {
                "input": data["posts"][i],
                "category": data["categories"][i],
                "intent": data["intents"][i],
                "implication": data["implications"][i],
                "target_group": data["target_groups"][i],
                "output": label,
            }
            for i in indices
        ]
    elif dataset_name == "semeval":
        return [
            {
                "input": data["posts"][i],
                "category": data["categories"][i],
                "aspect_terms": data["aspect_terms"][i],
                "polarity": data["polarities"][i],
                "output": label,
            }
            for i in indices
        ]
    elif dataset_name == "common_qa":
        return [
            {
                "input": data["posts"][i],
                "category": data["categories"][i],
                "concept": data["concept"][i],
                "distractor": data["distractor"][i],
                "output": label,
            }
            for i in indices
        ]
    elif dataset_name == "esnli":
        return [
            {
                "input": data["posts"][i],
                "category": data["categories"][i],
                "explanation": data["explanations"][i],
                "output": label,
            }
            for i in indices
        ]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")


def get_few_shot_examples(
    prompt_components: List[str],
    data: Dict[str, Dict[str, List[str]]],
    dataset_name: str,
    seed: int = RANDOM_SEED[0],
    all_categories: List[str] = None,
) -> List[Dict[str, str]]:
    """
    Generates few-shot examples for chain-of-thought reasoning.

    Args:
        prompt_components (List[str]): List of prompt components.
        data (Dict[str, Dict[str, List[str]]]): Dictionary containing the data.
        dataset_name (str): Name of the dataset.
        seed (int): Seed for random number generation.
        all_categories (Optional[List[str]]): List of all categories.

    Returns:
        List[Dict[str, str]]: List of generated examples.
    """
    random.seed(seed)

    if dataset_name == "esnli":
        pos_label = "contradiction"
        neg_label = "entailment"
    else:
        pos_label = Binary_bias_labels.POSITIVE.value
        neg_label = Binary_bias_labels.NEGATIVE.value

    results = []
    if "similar-few-shot" in prompt_components:
        N = NUM_OF_FEW_SHOT_EXAMPLES["similar_few_shot"]
        for example_group in data["similar_examples"]:
            local_examples = []
            for i in range(N):
                local_examples.append(
                    create_example(data, example_group["0"][i], neg_label, dataset_name)
                )
                local_examples.append(
                    create_example(data, example_group["1"][i], pos_label, dataset_name)
                )
            results.append(local_examples)

    elif "category-few-shot" in prompt_components:
        N = NUM_OF_FEW_SHOT_EXAMPLES["category_few_shot"]

        if not all_categories:
            all_categories = [member.value for member in SBIC_bias_categories]
        categories_with_index = sort_categories_by_index(
            data["categories"], all_categories
        )

        for curr_idx in range(len(data["posts"])):
            examples_ids = sample_from_each_category(categories_with_index, N, curr_idx)
            local_examples = []
            for cat, indices in examples_ids.items():
                label = neg_label if cat == NONE_CATEGORY_NAME else pos_label
                local_examples.extend(
                    create_examples(data, indices, label, dataset_name)
                )
            results.append(local_examples)

    elif "random-few-shot" in prompt_components:
        N = NUM_OF_FEW_SHOT_EXAMPLES["random_few_shot"]

        all_labels = list(set(label for label in data["labels"]))
        labels_with_index = sort_categories_by_index(data["labels"], all_labels)

        for curr_idx in range(len(data["posts"])):
            examples_ids = sample_from_each_category(labels_with_index, N, curr_idx)
            local_examples = []
            for label, indices in examples_ids.items():
                label_value = neg_label if label == 0 else pos_label
                local_examples.extend(
                    create_examples(data, indices, label_value, dataset_name)
                )
            results.append(local_examples)

    else:
        raise ValueError(f"No few shot type found in {prompt_components}")

    return [shuffle_list_with_seed(r, seed) for r in results]


def create_bias_prompts_from_base_template(
    prompt_components: List[str],
    data: Dict[str, Dict[str, List[str]]],
    description_type: str,
    definition_type: str,
    stimulus_type: str,
    reasoning_step_type: str = "",
    eos_token: str = "",
    return_format_example: BaseModel = None,
    cot_reasoning: List[str] = "",
    few_shot_mode: str = "default",
    seed: int = RANDOM_SEED[0],
    all_categories: List[str] = None,
    dataset_name: str = "sbic",
) -> List[str]:
    """
    Generates a list of prompts using a base template and various customizable components.

    Args:
        prompt_components (List[str]): List of components to include in the prompt.
        data (Dict[str, Dict[str, List[str]]]): Data containing the information needed to construct
            the prompts.
        description_type (str): Type of task description to use.
        definition_type (str): Specifies the type of definition to be included.
        stimulus_type (str): Specifies the type of stimulus to be included.
        reasoning_step_type (str): Specifies the type of reasoning step to be included.
        eos_token (str): Model specific end-of-sentence token.
        return_format_example (BaseModel): The format of the model response as pydantic model.
        cot_reasoning (List[str]): The previously generated reasoning for the Chain of Thought.
        few_shot_mode (str): The mode for few-shot sampling. 'default' or 'chain-of-thought'.
        seed (int): Seed for shuffling the few-shot examples.
        all_categories (List[str]): List of all categories to use for few-shot sampling.
        dataset_name (str): Name of the dataset.

    Returns:
        List[str]: A list of complete prompts.
    """
    if not isinstance(prompt_components, list):
        prompt_components = list(prompt_components)

    # read all necessary components from their respective JSON file
    components = {
        component: read_json(PROMPT_COMPONENTS_PATHS[component])
        for component in prompt_components
        if component in PROMPT_COMPONENTS_PATHS.keys()
    }

    # add these components separately since they are not in the components parsed from command line
    components["reasoning_step"] = read_json(PROMPT_COMPONENTS_PATHS["reasoning-steps"])
    components["task_description"] = read_json(
        PROMPT_COMPONENTS_PATHS["task-descriptions"]
    )

    prompt_params = {
        "eos_token": eos_token,
        "return_format_example": return_format_example,
        "task_description": components["task_description"].get(description_type, ""),
        "reasoning_step": components["reasoning_step"].get(reasoning_step_type, ""),
        "definition": components.get("definitions", {}).get(definition_type, ""),
        "directional_stimulus": components.get("directional-stimulus", {}).get(
            stimulus_type, ""
        ),
    }

    # create prompts that only contain the input text and the task description (special case)
    if "task_descriptions_only" in prompt_components:
        return [
            base_prompt(post, prompt_params["task_description"])
            for post in data["posts"]
        ]

    # generate few-shot examples if the prompt components include any few-shot components
    # otherwise generate a placeholder list of empty examples
    if set(FEW_SHOT_COMPONENTS).intersection(set(prompt_components)):
        # 'default' just concatenates the few-shot examples to the prompt
        if few_shot_mode == "default":
            few_shot_examples = get_few_shot_examples(
                prompt_components, data, dataset_name, seed, all_categories
            )
        # 'chain-of-thought' generates the reasoning for the few-shot examples in a predefined order
        elif few_shot_mode == "chain-of-thought":
            raise NotImplementedError("Chain of Thought mode not implemented yet.")
        else:
            raise ValueError(f"Invalid few-shot mode: {few_shot_mode}")
    else:
        few_shot_examples = [[]] * len(data["posts"])

    if cot_reasoning:
        return [
            base_prompt(
                post, **prompt_params, few_shot_examples=examples, cot_output=reason
            )
            for post, reason, examples in zip(
                data["posts"], cot_reasoning, few_shot_examples
            )
        ]

    return [
        base_prompt(post, **prompt_params, few_shot_examples=examples)
        for post, examples in zip(data["posts"], few_shot_examples)
    ]


def create_commonsense_prompts_from_base_template(
    prompt_components: List[str],
    data: Dict[str, Dict[str, List[str]]],
    description_type: str,
    definition_type: str,
    stimulus_type: str,
    reasoning_step_type: str = "",
    eos_token: str = "",
    return_format_example: BaseModel = None,
    cot_reasoning: List[str] = "",
    few_shot_mode: str = "default",
    seed: int = RANDOM_SEED[0],
    dataset_name: str = "common_qa",
    all_categories: List[str] = ["general"],
) -> List[str]:
    """
    Generates a list of prompts using a base template and various customizable components.

    Args:
        prompt_components (List[str]): List of components to include in the prompt.
        data (Dict[str, Dict[str, List[str]]]): Data containing the information needed to construct
            the prompts.
        description_type (str): Type of task description to use.
        definition_type (str): Specifies the type of definition to be included.
        stimulus_type (str): Specifies the type of stimulus to be included.
        reasoning_step_type (str): Specifies the type of reasoning step to be included.
        eos_token (str): Model specific end-of-sentence token.
        return_format_example (BaseModel): The format of the model response as pydantic model.
        cot_reasoning (List[str]): The previously generated reasoning for the Chain of Thought.
        few_shot_mode (str): The mode for few-shot sampling. 'default' or 'chain-of-thought'.
        seed (int): Seed for shuffling the few-shot examples.

    Returns:
        List[str]: A list of complete prompts.
    """
    if not isinstance(prompt_components, list):
        prompt_components = list(prompt_components)

    # read all necessary components from their respective JSON file
    components = {
        component: read_json(PROMPT_COMPONENTS_PATHS[component])
        for component in prompt_components
        if component in PROMPT_COMPONENTS_PATHS.keys()
    }

    # add these components separately since they are not in the components parsed from command line
    components["reasoning_step"] = read_json(PROMPT_COMPONENTS_PATHS["reasoning-steps"])
    components["task_description"] = read_json(
        PROMPT_COMPONENTS_PATHS["task-descriptions"]
    )

    prompt_params = {
        "eos_token": eos_token,
        "return_format_example": return_format_example,
        "task_description": components["task_description"].get(description_type, ""),
        "reasoning_step": components["reasoning_step"].get(reasoning_step_type, ""),
        "definition": components.get("definitions", {}).get(definition_type, ""),
        "directional_stimulus": components.get("directional-stimulus", {}).get(
            stimulus_type, ""
        ),
    }

    # create prompts that only contain the input text and the task description (special case)
    if "task_descriptions_only" in prompt_components:
        return [
            common_base_prompt(post, prompt_params["task_description"])
            for post in data["posts"]
        ]

    # generate few-shot examples if the prompt components include any few-shot components
    # otherwise generate a placeholder list of empty examples
    if set(FEW_SHOT_COMPONENTS).intersection(set(prompt_components)):
        # 'default' just concatenates the few-shot examples to the prompt
        if few_shot_mode == "default":
            few_shot_examples = get_few_shot_examples(
                prompt_components, data, dataset_name, seed, all_categories
            )
        # 'chain-of-thought' generates the reasoning for the few-shot examples in a predefined order
        elif few_shot_mode == "chain-of-thought":
            raise NotImplementedError("Chain of Thought mode not implemented yet.")
        else:
            raise ValueError(f"Invalid few-shot mode: {few_shot_mode}")
    else:
        few_shot_examples = [[]] * len(data["posts"])

    if cot_reasoning:
        return [
            common_base_prompt(
                post, **prompt_params, few_shot_examples=examples, cot_output=reason
            )
            for post, reason, examples in zip(
                data["posts"], cot_reasoning, few_shot_examples
            )
        ]

    return [
        common_base_prompt(post, **prompt_params, few_shot_examples=examples)
        for post, examples in zip(data["posts"], few_shot_examples)
    ]


def create_esnli_prompts_from_base_template(
    prompt_components: List[str],
    data: Dict[str, Dict[str, List[str]]],
    description_type: str,
    definition_type: str,
    stimulus_type: str,
    reasoning_step_type: str = "",
    eos_token: str = "",
    return_format_example: BaseModel = None,
    cot_reasoning: List[str] = "",
    few_shot_mode: str = "default",
    seed: int = RANDOM_SEED[0],
    dataset_name: str = "esnli",
    all_categories: List[str] = ["general"],
) -> List[str]:
    """
    Generates a list of prompts using a base template and various customizable components.

    Args:
        prompt_components (List[str]): List of components to include in the prompt.
        data (Dict[str, Dict[str, List[str]]]): Data containing the information needed to construct
            the prompts.
        description_type (str): Type of task description to use.
        definition_type (str): Specifies the type of definition to be included.
        stimulus_type (str): Specifies the type of stimulus to be included.
        reasoning_step_type (str): Specifies the type of reasoning step to be included.
        eos_token (str): Model specific end-of-sentence token.
        return_format_example (BaseModel): The format of the model response as pydantic model.
        cot_reasoning (List[str]): The previously generated reasoning for the Chain of Thought.
        few_shot_mode (str): The mode for few-shot sampling. 'default' or 'chain-of-thought'.
        seed (int): Seed for shuffling the few-shot examples.

    Returns:
        List[str]: A list of complete prompts.
    """
    if not isinstance(prompt_components, list):
        prompt_components = list(prompt_components)

    # read all necessary components from their respective JSON file
    components = {
        component: read_json(PROMPT_COMPONENTS_PATHS[component])
        for component in prompt_components
        if component in PROMPT_COMPONENTS_PATHS.keys()
    }

    # add these components separately since they are not in the components parsed from command line
    components["reasoning_step"] = read_json(PROMPT_COMPONENTS_PATHS["reasoning-steps"])
    components["task_description"] = read_json(
        PROMPT_COMPONENTS_PATHS["task-descriptions"]
    )

    prompt_params = {
        "eos_token": eos_token,
        "return_format_example": return_format_example,
        "task_description": components["task_description"].get(description_type, ""),
        "reasoning_step": components["reasoning_step"].get(reasoning_step_type, ""),
        "definition": components.get("definitions", {}).get(definition_type, ""),
        "directional_stimulus": components.get("directional-stimulus", {}).get(
            stimulus_type, ""
        ),
    }

    # create prompts that only contain the input text and the task description (special case)
    if "task_descriptions_only" in prompt_components:
        return [
            esnli_base_prompt(post, prompt_params["task_description"])
            for post in data["posts"]
        ]

    # generate few-shot examples if the prompt components include any few-shot components
    # otherwise generate a placeholder list of empty examples
    if set(FEW_SHOT_COMPONENTS).intersection(set(prompt_components)):
        # 'default' just concatenates the few-shot examples to the prompt
        if few_shot_mode == "default":
            few_shot_examples = get_few_shot_examples(
                prompt_components, data, dataset_name, seed, all_categories
            )
        # 'chain-of-thought' generates the reasoning for the few-shot examples in a predefined order
        elif few_shot_mode == "chain-of-thought":
            raise NotImplementedError("Chain of Thought mode not implemented yet.")
        else:
            raise ValueError(f"Invalid few-shot mode: {few_shot_mode}")
    else:
        few_shot_examples = [[]] * len(data["posts"])

    if cot_reasoning:
        return [
            esnli_base_prompt(
                post, **prompt_params, few_shot_examples=examples, cot_output=reason
            )
            for post, reason, examples in zip(
                data["posts"], cot_reasoning, few_shot_examples
            )
        ]

    return [
        esnli_base_prompt(post, **prompt_params, few_shot_examples=examples)
        for post, examples in zip(data["posts"], few_shot_examples)
    ]


def create_bias_prompts_from_cot_template(
    prompt_components: List[str],
    data: Dict[str, Dict[str, List[str]]],
    description_type: str,
    definition_type: str,
    stimulus_type: str,
    reasoning_step_type: str,
    eos_token: str = "",
    return_format_example: BaseModel = None,
    cot_reasoning: List[str] = "",
    seed: int = RANDOM_SEED[0],
    all_categories: List[str] = None,
    dataset_name: str = "sbic",
) -> List[str]:
    """
    Creates bias prompts from a chain of thought (CoT) template based on provided components and
    data.

    Args:
        prompt_components (List[str]): List of components to include in the prompt.
        data (Dict[str, Dict[str, List[str]]]): Dictionary containing the data for prompts. Should
        include 'posts'.
        description_type (str): Type of task description to use.
        definition_type (str): Type of definition to include in the prompt.
        stimulus_type (str): Type of stimulus to use in the prompt.
        reasoning_step_type (str): Type of reasoning step to include.
        eos_token (str, optional): End-of-sequence token to append to the prompt. Defaults to "".
        return_format_example (BaseModel, optional): Example of the format to return. Defaults to
        None.
        cot_reasoning (List[str], optional): List of chain-of-thought reasoning steps. Defaults to
        "".
        seed (int, optional): Seed for random number generation to ensure reproducibility. Defaults
        to RANDOM_SEED[0].
        all_categories (List[str], optional): List of all categories to consider. Defaults to None.
        dataset_name (str, optional): Name of the dataset to use ('sbic' or 'stereoset'). Defaults
        to "sbic".

    Raises:
        NotImplementedError: If the specified dataset is not implemented.

    Returns:
        List[str]: List of generated prompts.
    """
    if not isinstance(prompt_components, list):
        prompt_components = list(prompt_components)

    # select templates
    if dataset_name == "sbic":
        reasoning_prompt_template = sbic_chain_of_thought_prompt_reasoning
        prediction_prompt_template = sbic_chain_of_thought_prompt_prediction
    elif dataset_name == "stereoset":
        reasoning_prompt_template = stereoset_chain_of_thought_prompt_reasoning
        prediction_prompt_template = stereoset_chain_of_thought_prompt_prediction
    elif dataset_name == "cobra":
        reasoning_prompt_template = cobra_chain_of_thought_prompt_reasoning
        prediction_prompt_template = cobra_chain_of_thought_prompt_prediction
    elif dataset_name == "common_qa":
        reasoning_prompt_template = common_chain_of_thought_prompt_reasoning
        prediction_prompt_template = common_chain_of_thought_prompt_prediction
    elif dataset_name == "esnli":
        reasoning_prompt_template = esnli_chain_of_thought_prompt_reasoning
        prediction_prompt_template = esnli_chain_of_thought_prompt_prediction
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    # read all necessary components from their respective JSON file
    components = {
        component: read_json(PROMPT_COMPONENTS_PATHS[component])
        for component in prompt_components
        if component in PROMPT_COMPONENTS_PATHS.keys()
    }

    # add these components separately since they are not in the components parsed from command line
    components["reasoning_step"] = read_json(PROMPT_COMPONENTS_PATHS["reasoning-steps"])
    components["task_description"] = read_json(
        PROMPT_COMPONENTS_PATHS["task-descriptions"]
    )

    prompt_params = {
        "eos_token": eos_token,
        "return_format_example": return_format_example,
        "task_description": components["task_description"].get(description_type, ""),
        "reasoning_step": components["reasoning_step"].get(reasoning_step_type, ""),
        "definition": components.get("definitions", {}).get(definition_type, ""),
        "directional_stimulus": components.get("directional-stimulus", {}).get(
            stimulus_type, ""
        ),
    }

    # create prompts that only contain the input text and the task description (special case)
    if "task_descriptions_only" in prompt_components:
        if cot_reasoning:
            return [
                prediction_prompt_template(
                    post,
                    prompt_params["task_description"],
                    prompt_params["reasoning_step"],
                    reason,
                )
                for post, reason in zip(data["posts"], cot_reasoning)
            ]
        else:
            return [
                reasoning_prompt_template(
                    post,
                    prompt_params["task_description"],
                    prompt_params["reasoning_step"],
                )
                for post in data["posts"]
            ]

    # generate few-shot examples if the prompt components include any few-shot components
    # otherwise generate a placeholder list of empty examples
    if set(FEW_SHOT_COMPONENTS).intersection(set(prompt_components)):
        few_shot_examples = get_few_shot_examples(
            prompt_components, data, dataset_name, seed, all_categories
        )
    else:
        few_shot_examples = [[]] * len(data["posts"])

    if cot_reasoning:
        return [
            prediction_prompt_template(
                post, **prompt_params, few_shot_examples=examples, cot_reasoning=reason
            )
            for post, reason, examples in zip(
                data["posts"], cot_reasoning, few_shot_examples
            )
        ]

    return [
        reasoning_prompt_template(post, **prompt_params, few_shot_examples=examples)
        for post, examples in zip(data["posts"], few_shot_examples)
    ]


def get_system_prompt(
    prompt_components: List[str], system_prompt_type: str, system_prompt_name: str
) -> Union[str, None]:
    """
    Retrieves a system prompt based on the specified type and name.

    Args:
        prompt_components (List[str]): A list of prompt components to use.
        system_prompt_type (str): The category of the system prompt. 'personas' or 'behavior'.
        system_prompt_name (str): The specific name of the system prompt.

    Returns:
        Union[str, None]: The text of the system prompt if found, otherwise None.
    """

    if "system-prompts" in prompt_components:
        return (
            read_json(PROMPT_COMPONENTS_PATHS.get("system-prompts", {}))
            .get(system_prompt_type, {})
            .get(system_prompt_name, None)
        )
    return ""


def get_model_predictions(
    model_output: ModelOutput, labels: List[str]
) -> List[Dict[str, float]]:
    """
    Calculate the summed probabilities for each specified label based on the
    first token's probabilities from a list of token probability dictionaries.

    Args:
        model_output (ModelOutput)): The model output containing the token probabilities.
        labels (List[str]): A list of labels for which probabilities are to be summed.
            The function matches these labels case-insensitively to the tokens.

    Returns:
        List[Dict[str, float]]: A list of dictionaries where each dictionary contains
            the labels as keys and their corresponding summed probabilities as values.
    """
    output_token_probabilities = model_output.token_probabilities

    predictions = []
    for probabilities in output_token_probabilities:
        label_probabilities = {label: 0.0 for label in labels}

        for token, prob in probabilities[0].items():  # only first token is considered
            # token probabilities are sorted in descending order
            if prob == 0.0:
                break

            if token == "":
                continue

            for label in labels:
                if token.lower() in label.lower():
                    label_probabilities[label] += prob

        predictions.append(label_probabilities)

    return predictions


def create_component_combinations(args: argparse.Namespace) -> List[Tuple[str]]:
    """
    Generate combinations of given components with optional inclusion of few-shot sampling methods
    and a default task description.

    Args:
        args (argparse.Namespace): Command line arguments containing the components to combine.

    Returns:
        List[Tuple[str]]: A list of all possible combinations of the input lists. Each inner list
            is a tuple containing the components for a single combination.
    Raises:
        ValueError: If an element in the combinations is not recognized as valid.
    """

    def combine_list_elements(input_list: List[str]) -> List[Tuple[str]]:
        """Generate all possible combinations of the elements in the input list."""
        combinations = []
        for i in range(1, len(input_list) + 1):
            combinations.extend(list(itertools.combinations(input_list, i)))
        return combinations

    if args.few_shot_sampling_methods:
        combinations = []
        for method in args.few_shot_sampling_methods:
            new_components = args.prompt_components.copy()
            new_components.append(method)
            combinations.extend(combine_list_elements(new_components))
    else:
        combinations = combine_list_elements(args.prompt_components)

    # add the base case for the prompt (only task description)
    combinations.append(("task-description-only",))

    # last check for valid prompt components
    for combination in combinations:
        for element in combination:
            if (
                element not in VALID_PROMPT_COMPONENTS
                and element not in FEW_SHOT_COMPONENTS
            ):
                raise ValueError(
                    f"Invalid: {element} | valid: {VALID_PROMPT_COMPONENTS+FEW_SHOT_COMPONENTS}"
                )

    return list(set(combinations))


def get_predicted_labels(predictions: List[Dict[str, float]]) -> List[int]:
    """Generate binary predictions from label probability dictionaries.

    Args:
        predictions (List[Dict[str, float]]): A list of dictionaries where the keys are labels and
            the values are probabilities.

    Returns:
        List[int]: A list of binary predictions, where 1 indicates a positive label and 0 indicates
            non-positive label.
    """
    pos_label = Binary_bias_labels.POSITIVE.value

    predicted_labels = []
    for prob in predictions:
        prediction = 1 if max(prob, key=prob.get) == pos_label else 0
        predicted_labels.append(prediction)

    return predicted_labels


def should_skip_combination(
    result_path: str, combination: List[str]
) -> Tuple[bool, List[int]]:
    """Check if the combination should be skipped based on existing results.

    Args:
        result_path (str): Path to the result file.
        combination (List[str]): Combination of prompt components to check.

    Returns:
        Tuple[bool, List[int]]: A tuple containing a boolean indicating whether the combination
            should be skipped and a list of seeds that should be skipped.
    """
    seeds_to_skip = []

    if os.path.exists(result_path):
        if not set(FEW_SHOT_COMPONENTS).intersection(set(combination)):
            print(f"Results for {sorted(combination)} already exist. Skipping...")
            return True, seeds_to_skip

        with open(result_path, "r") as f:
            data_to_check = json.load(f)

        if all(str(key) in data_to_check for key in RANDOM_SEED):
            print(f"Results for {sorted(combination)} with all seeds already exist.")
            return True, seeds_to_skip

        for seed in RANDOM_SEED:
            if str(seed) in data_to_check:
                seeds_to_skip.append(seed)

    return False, seeds_to_skip


def create_absa_prompts_from_base_template(
    prompt_components: List[str],
    data: Dict[str, Dict[str, List[str]]],
    description_type: str,
    definition_type: str,
    stimulus_type: str,
    reasoning_step_type: str = "",
    eos_token: str = "",
    return_format_example: BaseModel = None,
    cot_reasoning: List[str] = "",
    few_shot_mode: str = "default",
    seed: int = RANDOM_SEED[0],
) -> List[str]:
    """
    Generates ABSA (Aspect-Based Sentiment Analysis) prompts based on the provided template
    components and data.

    Args:
        prompt_components (List[str]): List of prompt component names to include.
        data (Dict[str, Dict[str, List[str]]]): Input data containing posts and related information.
        description_type (str): Type of task description to use.
        definition_type (str): Type of definition for the task.
        stimulus_type (str): Type of stimulus to guide the prompt.
        reasoning_step_type (str, optional): Reasoning step type for chain-of-thought prompts.
            Defaults to "".
        eos_token (str, optional): End-of-sequence token. Defaults to "".
        return_format_example (BaseModel, optional): Example of the return format. Defaults to None.
        cot_reasoning (List[str], optional): List of chain-of-thought reasoning examples.
            Defaults to "".
        few_shot_mode (str, optional): Mode for generating few-shot examples
            ("default", "chain-of-thought"). Defaults to "default".
        seed (int, optional): Seed for randomization. Defaults to RANDOM_SEED[0].

    Raises:
        NotImplementedError: Raised if unsupported few-shot mode is provided.
        ValueError: Raised if an invalid few-shot mode or missing component is detected.

    Returns:
        List[str]: List of generated prompts.
    """
    if not isinstance(prompt_components, list):
        prompt_components = list(prompt_components)

    # read all necessary components from their respective JSON file
    components = {
        component: read_json(PROMPT_COMPONENTS_PATHS[component])
        for component in prompt_components
        if component in PROMPT_COMPONENTS_PATHS.keys()
    }

    # add these components separately since they are not in the components parsed from command line
    components["reasoning_step"] = read_json(PROMPT_COMPONENTS_PATHS["reasoning-steps"])
    components["task_description"] = read_json(
        PROMPT_COMPONENTS_PATHS["task-descriptions"]
    )

    prompt_params = {
        "eos_token": eos_token,
        "return_format_example": return_format_example,
        "task_description": components["task_description"].get(description_type, ""),
        "reasoning_step": components["reasoning_step"].get(reasoning_step_type, ""),
        "definition": components.get("definitions", {}).get(definition_type, ""),
        "directional_stimulus": components.get("directional-stimulus", {}).get(
            stimulus_type, ""
        ),
    }

    # get the review topic for the stimulus and format it
    if prompt_params["directional_stimulus"]:
        stimulus_texts = []
        categories = data["categories"]

        for cat in categories:
            stimulus = {"category": prompt_params["directional_stimulus"].format(cat)}
            stimulus_texts.append(stimulus)
    else:
        stimulus_texts = [{}] * len(data["posts"])

    # create prompts that only contain the input text and the task description (special case)
    if "task_descriptions_only" in prompt_components:
        return [
            absa_prompt(post, prompt_params["task_description"])
            for post in data["posts"]
        ]

    # generate few-shot examples if the prompt components include any few-shot components
    # otherwise generate a placeholder list of empty examples
    if set(FEW_SHOT_COMPONENTS).intersection(set(prompt_components)):
        # 'default' just concatenates the few-shot examples to the prompt
        if few_shot_mode == "default":
            few_shot_examples = get_absa_few_shot_examples(
                prompt_components, data, seed
            )
        # 'chain-of-thought' generates the reasoning for the few-shot examples in a predefined order
        elif few_shot_mode == "chain-of-thought":
            raise NotImplementedError("Chain of Thought mode not implemented yet.")
        else:
            raise ValueError(f"Invalid few-shot mode: {few_shot_mode}")
    else:
        few_shot_examples = [[]] * len(data["posts"])

    if cot_reasoning:
        return [
            absa_prompt(
                post, **prompt_params, few_shot_examples=examples, cot_output=reason
            )
            for post, reason, examples in zip(
                data["posts"], cot_reasoning, few_shot_examples
            )
        ]

    results = []
    for idx, (post, examples) in enumerate(zip(data["posts"], few_shot_examples)):
        results.append(
            absa_prompt(
                post,
                **prompt_params,
                few_shot_examples=examples,
                stimulus_text=stimulus_texts[idx],
            )
        )
    return results


def get_absa_few_shot_examples(
    prompt_components: List[str],
    data: Dict[str, Dict[str, List[str]]],
    seed: int = RANDOM_SEED[0],
) -> List[Dict[str, str]]:
    """
    Generates few-shot examples for ABSA tasks based on different prompt components.

    Args:
        prompt_components (List[str]): List of components determining the few-shot strategy.
        data (Dict[str, Dict[str, List[str]]]): Input data containing posts and examples.
        seed (int, optional): Seed for randomization of examples. Defaults to RANDOM_SEED[0].

    Raises:
        ValueError: Raised if no valid few-shot type is found in the provided components.

    Returns:
        List[Dict[str, str]]: List of few-shot examples for each post.
    """
    random.seed(seed)

    dataset_name = "semeval"
    pos_label, neg_label = "positive", "negative"
    all_categories = ["laptop", "restaurant"]

    results = []
    if "similar-few-shot" in prompt_components:
        N = NUM_OF_FEW_SHOT_EXAMPLES["similar_few_shot"]
        for example_group in data["similar_examples"]:
            local_examples = []
            for i in range(N):
                local_examples.append(
                    create_example(data, example_group["0"][i], neg_label, dataset_name)
                )
                local_examples.append(
                    create_example(data, example_group["1"][i], pos_label, dataset_name)
                )
            results.append(local_examples)

    elif "category-few-shot" in prompt_components:
        N = 4

        categories_with_index = sort_categories_by_index(
            data["categories"], all_categories
        )

        for curr_idx in range(len(data["posts"])):
            examples_ids = sample_from_each_category(categories_with_index, N, curr_idx)
            local_examples = []
            for cat, indices in examples_ids.items():
                label = neg_label if cat == NONE_CATEGORY_NAME else pos_label
                local_examples.extend(
                    create_examples(data, indices, label, dataset_name)
                )
            results.append(local_examples)

    elif "random-few-shot" in prompt_components:
        N = NUM_OF_FEW_SHOT_EXAMPLES["random_few_shot"]

        all_labels = list(set(label for label in data["labels"]))
        labels_with_index = sort_categories_by_index(data["labels"], all_labels)

        for curr_idx in range(len(data["posts"])):
            examples_ids = sample_from_each_category(labels_with_index, N, curr_idx)
            local_examples = []
            for label, indices in examples_ids.items():
                label_value = neg_label if label == 0 else pos_label
                local_examples.extend(
                    create_examples(data, indices, label_value, dataset_name)
                )
            results.append(local_examples)

    else:
        raise ValueError(f"No few shot type found in {prompt_components}")

    return [shuffle_list_with_seed(r, seed) for r in results]


def create_absa_prompts_from_cot_template(
    prompt_components: List[str],
    data: Dict[str, Dict[str, List[str]]],
    description_type: str,
    definition_type: str,
    stimulus_type: str,
    reasoning_step_type: str,
    eos_token: str = "",
    return_format_example: BaseModel = None,
    cot_reasoning: List[str] = "",
    seed: int = RANDOM_SEED[0],
):
    """
    Generates Aspect-Based Sentiment Analysis (ABSA) prompts using a chain-of-thought (CoT)
    template based on the provided components and data.

    Args:
        prompt_components (List[str]): List of component names to include in the prompt.
        data (Dict[str, Dict[str, List[str]]]): Dictionary containing the input data for the task.
        description_type (str): Type of task description to include in the prompt.
        definition_type (str): Type of definition to include in the prompt.
        stimulus_type (str): Type of stimulus or context used in the prompt.
        reasoning_step_type (str): Type of reasoning step to be included in the prompt.
        eos_token (str, optional): End-of-sequence token to use for the model. Defaults to "".
        return_format_example (BaseModel, optional): Format for the return example.
            Defaults to None.
        cot_reasoning (List[str], optional): Pre-generated chain-of-thought reasoning steps.
            Defaults to an empty string.
        seed (int, optional): Random seed for reproducibility of few-shot examples. Defaults to
            RANDOM_SEED[0].

    Returns:
        List[str]: A list of generated ABSA prompts, including the task description, stimulus,
        few-shot examples, and reasoning steps.
    """
    if not isinstance(prompt_components, list):
        prompt_components = list(prompt_components)

    # read all necessary components from their respective JSON file
    components = {
        component: read_json(PROMPT_COMPONENTS_PATHS[component])
        for component in prompt_components
        if component in PROMPT_COMPONENTS_PATHS.keys()
    }

    # add these components separately since they are not in the components parsed from command line
    components["reasoning_step"] = read_json(PROMPT_COMPONENTS_PATHS["reasoning-steps"])
    components["task_description"] = read_json(
        PROMPT_COMPONENTS_PATHS["task-descriptions"]
    )

    prompt_params = {
        "eos_token": eos_token,
        "return_format_example": return_format_example,
        "task_description": components["task_description"].get(description_type, ""),
        "reasoning_step": components["reasoning_step"].get(reasoning_step_type, ""),
        "definition": components.get("definitions", {}).get(definition_type, ""),
        "directional_stimulus": components.get("directional-stimulus", {}).get(
            stimulus_type, ""
        ),
    }

    # get the review topic for the stimulus and format it
    if prompt_params["directional_stimulus"]:
        stimulus_texts = []
        categories = data["categories"]

        for cat in categories:
            stimulus = {"category": prompt_params["directional_stimulus"].format(cat)}
            stimulus_texts.append(stimulus)
    else:
        stimulus_texts = [{}] * len(data["posts"])

    # create prompts that only contain the input text and the task description (special case)
    if "task_descriptions_only" in prompt_components:
        if cot_reasoning:
            return [
                semeval_chain_of_thought_prompt_prediction(
                    input=post,
                    task_description=prompt_params["task_description"],
                    reasoning_step=prompt_params["reasoning_step"],
                    cot_reasoning=reason,
                )
                for post, reason in zip(data["posts"], cot_reasoning)
            ]
        else:
            return [
                semeval_chain_of_thought_prompt_reasoning(
                    input=post,
                    task_description=prompt_params["task_description"],
                    reasoning_step=prompt_params["reasoning_step"],
                )
                for post in data["posts"]
            ]

    # generate few-shot examples if the prompt components include any few-shot components
    # otherwise generate a placeholder list of empty examples
    if set(FEW_SHOT_COMPONENTS).intersection(set(prompt_components)):
        few_shot_examples = get_absa_few_shot_examples(prompt_components, data, seed)
    else:
        few_shot_examples = [[]] * len(data["posts"])

    results = []
    if cot_reasoning:
        for idx, (post, examples) in enumerate(zip(data["posts"], few_shot_examples)):
            prompt = semeval_chain_of_thought_prompt_prediction(
                post,
                **prompt_params,
                few_shot_examples=examples,
                cot_reasoning=cot_reasoning[idx],
                stimulus_text=stimulus_texts[idx],
            )
            results.append(prompt)
    else:
        for idx, (post, examples) in enumerate(zip(data["posts"], few_shot_examples)):
            prompt = semeval_chain_of_thought_prompt_reasoning(
                post,
                **prompt_params,
                few_shot_examples=examples,
                stimulus_text=stimulus_texts[idx],
            )
            results.append(prompt)
    return results
