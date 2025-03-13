import argparse
import json
import random
from datetime import datetime
from os import makedirs, path

import evaluate
import numpy as np
import pandas as pd
import torch
from icecream import ic
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed,
)
from transformers.pipelines.pt_utils import KeyDataset

from datasets import Dataset
from src.bias_detection.config.settings import FILE_NAME_TEMPLATES, RANDOM_SEED
from src.bias_detection.data_handler import DataHandler
from src.bias_detection.utils.general import get_custom_timestamp_string

# from datasets import Dataset

ic.configureOutput(prefix=get_custom_timestamp_string)

# ##############################################
# Hyperparameters
BATCH_SIZE = 64
BATCH_SIZE_VAL = 32
# BATCH_SIZE_TEST = 32
MAX_SEQUENCE_LENGTH = 256
TRAIN_EPOCHS = 20
# Evaluate (roughly) n times for a data length of 5k instances
EVAL_STEPS = int((TRAIN_EPOCHS * (5000 / BATCH_SIZE)) / 5)
# TRAIN_EPOCHS = 1

MODEL_BASE_NAME = ""
LOG_DIR = ""


def main(args):
    global MODEL_BASE_NAME, LOG_DIR
    MODEL_BASE_NAME = path.basename(args.model_name)

    # Set seeds for reproducable results
    set_seed(seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ic(f">> Working on device '{device}'")

    # ##############################################################################################
    # Data loading and preparation
    prompt_components = [c.split(";") for c in args.prompt_components]

    ic(">> Loading data")

    # -----------------------
    # Training data
    ic("> Training data")
    # Collect all results for the provided model
    model_results = pd.DataFrame()
    for composition in prompt_components:
        if composition[0] == "cot":
            prefix = f"{args.data}/{args.data}-cot-greedy-train_"
        else:
            prefix = f"{args.data}/{args.data}-greedy-train_"
        # Load component specific data
        data_filename = FILE_NAME_TEMPLATES["prompt-predictions"].format(
            prefix=prefix,
            model_name=args.llm_name,
            components="_".join([i for i in composition if i != "cot"]),
        )

        # Load component-specific results
        component_results = pd.read_parquet(data_filename)

        # If there are no outputs for the provided seed, we fallback to the first seed in the seed
        # list; this might happen for predictions where a seed has no influence on the inference,
        # such as defintions
        if f"output_{args.seed}" in component_results:
            seed_to_use = args.seed
        else:
            seed_to_use = RANDOM_SEED[0]
            print(
                f"Seed '{args.seed}' not available for '{composition}'; "
                f"using '{RANDOM_SEED[0]}' instead."
            )

        # If this is the first step in the loop, we need to add the input texts to the results
        # dataframe
        if "input" not in model_results.columns:
            model_results["input"] = component_results["input"]

        # Infer prediction correctness (which represents the y-label for the prediction model)
        model_results["_".join(composition)] = (
            component_results["true_label"]
            == component_results[f"output_{seed_to_use}"]
        ).astype(int)

    # Create mapping from component to label index/id for all components collected from the result
    # files
    composition_list_train = model_results.loc[
        :, model_results.columns != "input"
    ].columns
    id2composition_str = {
        str(index): composition
        for index, composition in enumerate(composition_list_train)
    }
    id2composition = {
        index: composition for index, composition in enumerate(composition_list_train)
    }
    composition2id = {
        composition: index for index, composition in id2composition.items()
    }

    # For each instance, gather its index, the input text and label vectors (which potentially has
    # 1 in multiple positions)
    data_with_label_vectors_train = [
        {"index": i, "input": list(j)[0], "labels": np.array(list(j)[1:]).astype(float)}
        for i, j in model_results.iterrows()
    ]
    dataset_train = Dataset.from_list(data_with_label_vectors_train)
    if not len(data_with_label_vectors_train[0]["labels"]) == len(
        composition_list_train
    ):
        raise ValueError(
            "Number of components is not the same as the number of labels. This might happen if"
            "there was an error during parsing files for all components."
        )

    # ---------------------
    # Validation data
    ic("> Validation data")
    model_results_val = pd.DataFrame()
    for composition in prompt_components:
        if composition[0] == "cot":
            prefix = f"{args.data}/{args.data}-cot-greedy-dev_"
        else:
            prefix = f"{args.data}/{args.data}-greedy-dev_"
        # Load component specific data
        data_filename = FILE_NAME_TEMPLATES["prompt-predictions"].format(
            prefix=prefix,
            model_name=args.llm_name,
            components="_".join([i for i in composition if i != "cot"]),
        )

        # Load component-specific results
        component_results = pd.read_parquet(data_filename)

        # If there are no outputs for the provided seed, we fallback to the first seed in the seed
        # list; this might happen for predictions where a seed has no influence on the inference,
        # such as defintions
        if f"output_{args.seed}" in component_results:
            seed_to_use = args.seed
        else:
            seed_to_use = RANDOM_SEED[0]
            print(
                f"Seed '{args.seed}' not available for '{composition}'; "
                f"using '{RANDOM_SEED[0]}' instead."
            )

        # If this is the first step in the loop, we need to add the input texts to the results
        # dataframe
        if "input" not in model_results_val.columns:
            model_results_val["input"] = component_results["input"]

        # Infer prediction correctness (which represents the y-label for the prediction model)
        model_results_val["_".join(composition)] = (
            component_results["true_label"]
            == component_results[f"output_{seed_to_use}"]
        ).astype(int)

    # Create mapping from component to label index/id for all components collected from the result
    # files
    component_list_val = model_results_val.loc[
        :, model_results_val.columns != "input"
    ].columns

    # For each instance, gather its index, the input text and label vectors (which potentially has
    # 1 in multiple positions)
    data_with_label_vectors_val = [
        {"index": i, "input": list(j)[0], "labels": np.array(list(j)[1:]).astype(float)}
        for i, j in model_results_val.iterrows()
    ]
    dataset_val = Dataset.from_list(data_with_label_vectors_val)
    if not len(data_with_label_vectors_val[0]["labels"]) == len(component_list_val):
        raise ValueError(
            "Number of components is not the same as the number of labels. This might happen if"
            "there was an error during parsing files for all components."
        )
    if not (composition_list_train == component_list_val).all():
        raise ValueError(
            "Component lists of training and validation set seem to be different."
            "Make sure they are the same"
        )

    # ##############################################################################################
    # Model loading and definitions
    num_labels = len(composition_list_train)
    ic(f">> Working with {num_labels} labels")

    # Load the tokenizer and model
    ic(f">> Loading tokenizer from '{args.model_name}'")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess_function(instance):
        example = tokenizer(
            instance["input"],
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            padding="max_length",
        )
        example["labels"] = instance["labels"]
        return example

    tokenized_dataset_train = dataset_train.map(
        preprocess_function,
        batched=True,
        cache_file_name=f"/tmp/{args.data}-composition_prediction-train_{MODEL_BASE_NAME}-batch{BATCH_SIZE}.arrow",
        num_proc=12,
    )
    tokenized_dataset_val = dataset_val.map(
        preprocess_function,
        batched=True,
        cache_file_name=f"/tmp/{args.data}-composition_prediction-val_{MODEL_BASE_NAME}-batch{BATCH_SIZE_VAL}.arrow",
        num_proc=12,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ic(f">> Loading model from '{args.model_name}'")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2composition,
        label2id=composition2id,
        problem_type="multi_label_classification",
    )
    model.to(device)

    clf_metrics = evaluate.combine(
        [
            "./evaluate/metrics/accuracy/accuracy.py",
            "./evaluate/metrics/f1/f1.py",
            "./evaluate/metrics/precision/precision.py",
            "./evaluate/metrics/recall/recall.py",
        ]
    )

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = sigmoid(predictions)
        predictions = (predictions > 0.5).astype(int).reshape(-1)
        return clf_metrics.compute(
            predictions=predictions, references=labels.astype(int).reshape(-1)
        )

    model_path_basename = f"{MODEL_BASE_NAME}_{args.model_identifier}"

    training_args = TrainingArguments(
        logging_dir=path.join("logs", model_path_basename),
        logging_steps=5,
        output_dir=path.join("intermediate", f"{model_path_basename}-checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE_VAL,
        num_train_epochs=TRAIN_EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        greater_is_better=True,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    ic("==Saving best model")
    model_save_path = f"models/composition_prediction/{model_path_basename}"
    trainer.save_model(model_save_path)

    # Saving id/index composition map
    with open(path.join(model_save_path, "id2composition_map.json"), "w") as f:
        json.dump(id2composition_str, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help=(
            "Path or name to the model that will be fine-tuned as the composition"
            "prediction model."
        ),
        required=True,
    )
    parser.add_argument(
        "--llm-name",
        type=str,
        help="Name of the LLM which's predictions should be used for training and inference.",
        required=True,
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix to add to the result and log file name.",
        required=True,
    )
    parser.add_argument(
        "--prompt_components",
        type=str,
        nargs="+",
        help=(
            "A string that describes the prompt component for which the regression model is being "
            "trained. Here, the component is mostly important to find the correct training data "
            "file."
        ),
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "Seed for which to load the LLM prediction data. Also used to set the torch seed for"
            "training and inference of the composition prediction."
        ),
        required=True,
    )
    parser.add_argument(
        "--model_identifier",
        type=str,
        help=(
            "A unique string that is appended to the end of each model-specific file saved (i.e., "
            "logs, checkpoints, predictions). This makes it easier to train multiple models and "
            "evaluate them later."
        ),
        default="finetune",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Dataset for which to predict compositions.",
        required=True,
    )
    args = parser.parse_args()

    main(args)

    ic("Done.")
