import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel


def read_json(path: str) -> Dict[str, Any]:
    """
    Reads a JSON file from the given file path and returns its contents as a dictionary.

    Args:
        path (str): The file path to the JSON file that needs to be read.

    Raises:
        ValueError: Raised if the input `path` is not a string, indicating that the
            provided path is not valid.
        FileNotFoundError: Raised if no file exists at the given `path`, indicating the
            file could not be found or does not exist.

    Returns:
        Dict[str, Any]: A dictionary representing the JSON data.
    """
    if not isinstance(path, str):
        raise ValueError("The path must be a string")

    if not os.path.exists(path):
        raise FileNotFoundError(f"The file at path {path} does not exist")

    with open(path, "r") as f:
        json_data = json.load(f)

    return json_data


def save_dict_to_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Saves a dictionary to a JSON file with indentation and ensures the path exists.

    Args:
      data: The dictionary to save.
      file_path: The path to the JSON file.
    """
    _, ext = os.path.splitext(file_path)
    if not ext:
        file_path += ".json"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def save_string_to_txt(data: str, file_path: str) -> None:
    """
    Saves a string to a text file, overwriting the existing file if it already exists,
    and ensures the directory for the file path exists.

    Args:
      data: The string to save.
      file_path: The path to the text file.
    """
    _, ext = os.path.splitext(file_path)
    if not ext:
        file_path += ".txt"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Open the file in write mode ('w') which overwrites the file if it already exists
    with open(file_path, "w") as f:
        f.write(data)


def save_results_to_json(
    results: Dict[str, Any], file_path: str, write_mode: str = "a"
) -> None:
    """
    Saves the given results dictionary to a JSON file at the specified file path.

    Args:
        results (Dict[str, Any]): A dictionary with results.
        file_path (str): The path to the file where the results will be saved.
        write_mode (str, optional): Specifies the mode of writing to the file:
                                    'a' for append mode
                                    'w' for write mode

    Raises:
        ValueError: If the `write_mode` is not one of ['a', 'w'].
    """
    if write_mode not in ["a", "w"]:
        raise ValueError("The write mode must be either 'a' or 'w'")

    results = {str(key): value for key, value in results.items()}

    if write_mode == "a" and os.path.exists(file_path):
        existing_results = read_json(file_path)
        existing_results.update(results)
        results = existing_results

    save_dict_to_json(results, file_path)


def combine_results(
    seed: int, evaluation: Dict[str, float], outputs: List[str]
) -> Dict[str, Any]:
    """Function to combine the evaluation results and outputs into a single dictionary.

    Args:
        seed (int): Experiment seed.
        evaluation (Dict[str, float]): F1, precision, recall, and accuracy scores.
        outputs (List[str]): List of model outputs.

    Returns:
        Dict[str, Any]: Combined results dictionary.
    """
    combined_results = {
        seed: {
            "eval_results": evaluation,
            "output_stats": {
                "num_outputs": len(outputs),
                "output_set": sorted(list(set(outputs))),
                "output_distribution": dict(sorted(Counter(outputs).items())),
            },
        }
    }
    return combined_results


def save_df_to_csv(data: pd.DataFrame, file_path: str) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Args:
        data (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to the CSV file.
    """
    if not file_path.endswith(".csv"):
        file_path += ".csv"

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    data.to_csv(file_path, index=False)


def save_df_to_parquet(data: pd.DataFrame, file_path: str) -> None:
    """
    Save a pandas DataFrame to a parquet file.

    Args:
        data (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to the parquet file.
    """
    if not file_path.endswith(".parquet"):
        file_path += ".parquet"

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    data.to_parquet(file_path, index=False, engine="pyarrow")


def save_results_to_csv(df: pd.DataFrame, file_path: str, seed: int) -> None:
    """
    Saves or updates experiment results to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing the results of the experiment.
        file_path (str): The path to the CSV file where the experiment results are saved.
            If the file exists, it will be updated; otherwise, a new file will be created.
        seed (int): The seed value used in the experiment.
    """
    output_col = f"output_{seed}"
    output_prob_col = f"output_prob_dist_{seed}"

    if Path(file_path).exists():
        existing_df = pd.read_csv(file_path)

        existing_df[output_col] = df["output"]
        existing_df[output_prob_col] = df["output_prob_dist"]

        cols = sorted(
            existing_df.columns, key=lambda x: (x not in ["input", "true_label"], x)
        )
        existing_df = existing_df[cols]

        save_df_to_csv(existing_df, file_path)
    else:
        df.rename(
            columns={"output": output_col, "output_prob_dist": output_prob_col},
            inplace=True,
        )

        cols = sorted(df.columns, key=lambda x: (x not in ["input", "true_label"], x))
        df = df[cols]

        save_df_to_csv(df, file_path)


def ensure_correct_extension(file_path: str, file_type: str) -> str:
    """
    Ensures the correct file extension for the file_path based on the file_type.

    Args:
        file_path (str): The original file path.
        file_type (str): The expected file type ('csv' or 'parquet').

    Returns:
        str: The corrected file path with the proper extension.
    """
    expected_extension = file_type
    current_extension = file_path.split(".")[-1]

    if current_extension != expected_extension:
        file_path = f"{file_path.rsplit('.', 1)[0]}.{expected_extension}"

    return file_path


def save_results_to_file(
    df: pd.DataFrame, file_path: str, seed: int, file_type: str
) -> None:
    """
    Saves or updates experiment results to a specified file type (CSV or Parquet).

    Args:
        df (pd.DataFrame): DataFrame containing the results of the experiment.
        file_path (str): The path to the file where the experiment results are saved.
            If the file exists, it will be updated; otherwise, a new file will be created.
        seed (int): The seed value used in the experiment.
        file_type (str): The type of file to save the results to ('csv' or 'parquet').
    """
    output_col = f"output_{seed}"
    output_prob_col = f"output_prob_dist_{seed}"

    if file_type not in ["csv", "parquet"]:
        raise ValueError(
            f"file_type must be either 'csv' or 'parquet'. Found: {file_type}"
        )

    file_path = ensure_correct_extension(file_path, file_type)

    if Path(file_path).exists():
        if file_type == "csv":
            existing_df = pd.read_csv(file_path)
        else:
            existing_df = pd.read_parquet(file_path)

        existing_df[output_col] = df["output"]
        existing_df[output_prob_col] = df["output_prob_dist"]

        cols = sorted(
            existing_df.columns, key=lambda x: (x not in ["input", "true_label"], x)
        )
        existing_df = existing_df[cols]

        if file_type == "csv":
            save_df_to_csv(existing_df, file_path)
        else:
            save_df_to_parquet(existing_df, file_path)
    else:
        df.rename(
            columns={"output": output_col, "output_prob_dist": output_prob_col},
            inplace=True,
        )

        cols = sorted(df.columns, key=lambda x: (x not in ["input", "true_label"], x))
        df = df[cols]

        if file_type == "csv":
            save_df_to_csv(df, file_path)
        else:
            save_df_to_parquet(df, file_path)


def generate_custom_json_regex(
    model_class: type(BaseModel), field_constraints: Optional[Dict[str, str]] = None
) -> str:
    """
    Generates a regex pattern for matching JSON objects based on a Pydantic model's
    structure, with optional field constraints.

    Args:
        model_class (type(BaseModel)): The Pydantic model class to generate the
            regex for.
        field_constraints (Optional[Dict[str, str]]): Optional dictionary specifying
            regex constraints for certain fields, e.g., `{"first_name": "(Tom|Jerry)"}`.

    Returns:
        str: Regex pattern matching JSON objects as per the Pydantic model and
            field constraints.
    """
    json_object_parts = []

    for field_name, field_type in model_class.__fields__.items():
        if field_constraints and field_name in field_constraints:
            field_value_regex = field_constraints[field_name]
        else:
            field_value_regex = r'(?:[^"\\\\\\x00-\\x1f\\x7f-\\x9f]|\\\\.)*'

        field_regex = f'[\\n ]*"{field_name}"[\\n ]*:[\\n ]*"{field_value_regex}"'
        json_object_parts.append(field_regex)

    json_object_regex = "\\{" + ",".join(json_object_parts) + "[\\n ]*\\}"

    return json_object_regex


def get_custom_timestamp_string() -> str:
    """Return a nicely formatted datetime string that can, for example, be used for logging."""
    return "%s |> " % time.strftime("%Y-%m-%d--%H:%M:%S")


def shuffle_list_with_seed(input_list: List[Any], seed: int) -> List[Any]:
    """Shuffles a list using the given seed."""
    random.seed(seed)
    random.shuffle(input_list)
    return input_list


def create_directories(directories: Union[List[str], str]) -> None:
    """Create directories if they do not exist."""
    if isinstance(directories, str):
        directories = [directories]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def sort_categories_by_index(
    data: List[List[str]], all_categories: List[str], empty_category: str = "none"
) -> Dict[str, List[int]]:
    """Sort categories by their indices.

    Args:
        data (List[List[str]]): A list of lists where each sublist represents categories associated
            with a particular item.
        all_categories (List[str]): A list of all possible categories.
        empty_category (str, optional): The category to use when an item has no categories.
            Defaults to "none".

    Returns:
        Dict[str, List[int]]: Dictionary where keys are categories and values are lists of indices.
    """
    if isinstance(data[0], np.ndarray):
        data = [d.tolist() for d in data]

    if not isinstance(data[0], list):
        data = [[d] for d in data]

    for idx, d in enumerate(data):
        if not d:
            data[idx] = [empty_category]

    results = {category: [] for category in all_categories}
    for category in all_categories:
        for idx, d in enumerate(data):
            if category in d:
                results[category].append(idx)

    return results


def sample_from_each_category(
    data: Dict[str, List[int]], N: int, current_idx: int
) -> Dict[str, List[int]]:
    """Sample a specified number of items from each category.

    Args:
        data (Dict[str, List[int]]): A dictionary where keys are categories and values are lists of
            item indices.
        N (int): The maximum number of items to sample from each category.
        current_idx (int): The index of the item to exclude from sampling.

    Returns:
        Dict[str, List[int]]: A dictionary where keys are categories and values are lists of
            sampled item indices.
    """
    results = {category: [] for category in data.keys()}
    already_sampled = set([current_idx])

    for category, indices in data.items():
        possible_indices = list(set(indices) - already_sampled)

        samples = random.sample(possible_indices, N)  # N << len(possible_indices)

        already_sampled.update(samples)

        results[category] = samples

    return results


def safely_extract_str_from_json(
    json_list: List[Union[str, Dict[str, str]]], key: str, answer_choices: List[str]
) -> List[str]:
    """
    Safely extracts values associated with a specified key from a list of JSON objects.
    Handles both dictionaries and strings containing the possible answer choices.

    Args:
        json_list (List[Union[str, Dict[str, str]]]): A list of JSON objects, which can be either
            dictionaries or strings.
        key (str): The key whose value needs to be extracted from the JSON objects.
        answer_choices (List[str]): A list of possible answer choices.

    Returns:
        List[str]: A list of extracted values corresponding to the specified key or answer choices.
    """
    if not isinstance(json_list, list):
        json_list = [json_list]

    extracted_output = []
    for obj in json_list:
        if isinstance(obj, dict):
            extracted_output.append(obj[key])

        elif isinstance(obj, str):
            for choice in answer_choices:
                if choice in obj:
                    extracted_output.append(choice)
                    print("extracted:", choice)
                    break
        else:
            raise ValueError(
                "The input JSON list must contain dictionaries or strings."
                f"Found type({type(obj)}) var({obj})"
            )

    return extracted_output


def ceil_to_next_even(num: float) -> int:
    """
    Rounds a given float to the next even integer.

    Args:
        num (float): The number to be rounded up.

    Returns:
        int: The next even integer greater than or equal to the given number.
    """
    return math.ceil(num / 2) * 2


def process_jsonl(file_path: str) -> List[int]:
    """
    Reads a JSONL file, sorts the entries by `custom_id` in ascending order,
    converts the message content to binary values (0 for "no" and 1 for "yes"),
    and returns the binary values as a list of integers.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        List[int]: A list of binary values based on the message content.
    """
    try:
        # read and parse the JSONL file
        with open(file_path, "r") as file:
            data = [json.loads(line) for line in file]

        # convert message content to binary values and collect them in a list
        binary_values = []
        for entry in data:
            message_content = entry["response"]["body"]["choices"][0]["message"][
                "content"
            ]
            if "yes" in message_content:
                binary_values.append(1)
            else:
                binary_values.append(0)

        return binary_values

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing file: {e}")
        return []


def safe_get(obj, key, default="no"):
    """
    Safely retrieves a value from a dictionary-like object.

    Args:
        obj (dict): The dictionary-like object to retrieve the value from.
        key (hashable): The key to look up in the object.
        default (Any, optional): The value to return if the key is not found.
            Defaults to "no".

    Returns:
        Any: The value associated with the key if it exists in the object,
             otherwise the default value.

    Raises:
        TypeError: If the obj parameter is not a dictionary-like object that
                   supports key-based access.

    Example:
        >>> data = {"name": "Alice", "age": 30}
        >>> safe_get(data, "name")
        'Alice'
        >>> safe_get(data, "city", "Unknown")
        'Unknown'
    """
    try:
        return obj[key]
    except KeyError:
        return default
