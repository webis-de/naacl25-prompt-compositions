import argparse
import json
from os import makedirs, path

import numpy as np
import pandas as pd
import torch
from icecream import ic
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    set_seed,
)
from transformers.pipelines.pt_utils import KeyDataset

from datasets import Dataset
from src.bias_detection.data_handler import DataHandler

# Hyperparameters
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 32


def main(args):
    # Set seeds for reproducable results
    set_seed(seed=args.seed)
    MODEL_BASE_NAME = path.basename(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ic(f">> Working on device '{device}'")
    model_path_basename = f"{MODEL_BASE_NAME}_{args.model_identifier}"
    model_path = path.join(f"models/composition_prediction/{model_path_basename}")

    # Data loading
    # ----------------------------
    # Load the tokenizer and model
    ic(f">> Loading tokenizer from '{args.model_name}'")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    ic(">> Loading data")
    if args.data == "sbic":
        data_handler = DataHandler(datasets_to_load=["sbic"])

        ic("> Validation data")
        val_df = data_handler.sbic_data["dev"].rename(
            columns={"post": "text", "hasBiasedImplication": "labels"}
        )[["text", "labels"]]

        data_with_label_vectors_val = [
            {"index": i, "input": j["text"], "labels": j["labels"]}
            for i, j in val_df.iterrows()
        ]
        dataset_val = Dataset.from_list(data_with_label_vectors_val)

        ic("> Test data")
        test_df = data_handler.sbic_data["test"].rename(
            columns={"post": "text", "hasBiasedImplication": "labels"}
        )[["text", "labels"]]

        data_with_label_vectors_test = [
            {"index": i, "input": j["text"], "labels": j["labels"]}
            for i, j in test_df.iterrows()
        ]
        dataset_test = Dataset.from_list(data_with_label_vectors_test)
    elif args.data == "stereoset":
        data_handler = DataHandler(datasets_to_load=["stereoset"])

        ic("> Validation data")
        val_df = data_handler.stereoset_data["dev"][
            ["text", "hasBiasedImplication", "text_hash"]
        ].rename(columns={"hasBiasedImplication": "labels", "text_hash": "md5_hash"})

        data_with_label_vectors_val = [
            {"index": i, "input": j["text"], "labels": j["labels"]}
            for i, j in val_df.iterrows()
        ]
        dataset_val = Dataset.from_list(data_with_label_vectors_val)

        ic("> Test data")
        test_df = data_handler.stereoset_data["test"][
            ["text", "hasBiasedImplication", "text_hash"]
        ].rename(columns={"hasBiasedImplication": "labels", "text_hash": "md5_hash"})

        data_with_label_vectors_test = [
            {"index": i, "input": j["text"], "labels": j["labels"]}
            for i, j in test_df.iterrows()
        ]
        dataset_test = Dataset.from_list(data_with_label_vectors_test)
    elif args.data == "cobra_frames":
        data_handler = DataHandler(datasets_to_load=["cobra_frames"])

        ic("> Validation data")
        val_df = data_handler.cobra_frames["dev"][
            ["post", "hasBiasedImplication", "text_hash"]
        ].rename(
            columns={
                "text_hash": "md5_hash",
                "hasBiasedImplication": "labels",
                "post": "text",
            }
        )

        data_with_label_vectors_val = [
            {"index": i, "input": j["text"], "labels": j["labels"]}
            for i, j in val_df.iterrows()
        ]
        dataset_val = Dataset.from_list(data_with_label_vectors_val)

        ic("> Test data")
        test_df = data_handler.cobra_frames["test"][
            ["post", "hasBiasedImplication", "text_hash"]
        ].rename(
            columns={
                "text_hash": "md5_hash",
                "hasBiasedImplication": "labels",
                "post": "text",
            }
        )

        data_with_label_vectors_test = [
            {"index": i, "input": j["text"], "labels": j["labels"]}
            for i, j in test_df.iterrows()
        ]
        dataset_test = Dataset.from_list(data_with_label_vectors_test)
    elif args.data == "semeval":
        data_handler = DataHandler(datasets_to_load=["semeval"])

        ic("> Validation data")
        val_df = data_handler.semeval_data["dev"][["post", "label", "md5_hash"]].rename(
            columns={"label": "labels", "post": "text"}
        )

        data_with_label_vectors_val = [
            {"index": i, "input": j["text"], "labels": j["labels"]}
            for i, j in val_df.iterrows()
        ]
        dataset_val = Dataset.from_list(data_with_label_vectors_val)

        ic("> Test data")
        test_df = data_handler.semeval_data["test"][
            ["post", "label", "md5_hash"]
        ].rename(columns={"label": "labels", "post": "text"})

        data_with_label_vectors_test = [
            {"index": i, "input": j["text"], "labels": j["labels"]}
            for i, j in test_df.iterrows()
        ]
        dataset_test = Dataset.from_list(data_with_label_vectors_test)
    elif args.data == "esnli":
        data_handler = DataHandler(datasets_to_load=["esnli"])

        ic("> Validation data")
        val_df = data_handler.esnli_data["dev"][["post", "label", "md5_hash"]].rename(
            columns={"label": "labels", "post": "text"}
        )

        data_with_label_vectors_val = [
            {"index": i, "input": j["text"], "labels": j["labels"]}
            for i, j in val_df.iterrows()
        ]
        dataset_val = Dataset.from_list(data_with_label_vectors_val)

        ic("> Test data")
        test_df = data_handler.esnli_data["test"][["post", "label", "md5_hash"]].rename(
            columns={"label": "labels", "post": "text"}
        )

        data_with_label_vectors_test = [
            {"index": i, "input": j["text"], "labels": j["labels"]}
            for i, j in test_df.iterrows()
        ]
        dataset_test = Dataset.from_list(data_with_label_vectors_test)
    elif args.data == "commonsense_qa":
        data_handler = DataHandler(datasets_to_load=["common_qa"])

        ic("> Validation data")
        val_df = data_handler.common_qa["dev"]
        val_df["qa_concat"] = val_df.apply(
            lambda row: f"[Q] {row['question']} [A] {row['answer']}", axis=1
        )
        val_df = val_df[["qa_concat", "label", "md5_hash"]].rename(
            columns={"label": "labels", "qa_concat": "text"}
        )

        data_with_label_vectors_val = [
            {"index": i, "input": j["text"], "labels": j["labels"]}
            for i, j in val_df.iterrows()
        ]
        dataset_val = Dataset.from_list(data_with_label_vectors_val)

        ic("> Test data")
        test_df = data_handler.common_qa["test"]
        test_df["qa_concat"] = test_df.apply(
            lambda row: f"[Q] {row['question']} [A] {row['answer']}", axis=1
        )
        test_df = test_df[["qa_concat", "label", "md5_hash"]].rename(
            columns={"label": "labels", "qa_concat": "text"}
        )

        data_with_label_vectors_test = [
            {"index": i, "input": j["text"], "labels": j["labels"]}
            for i, j in test_df.iterrows()
        ]
        dataset_test = Dataset.from_list(data_with_label_vectors_test)
    else:
        raise ValueError("Dataset not implemented.")

    ic("> Composition map")
    with open(path.join(model_path, "id2composition_map.json"), "r") as f:
        id2composition_str = json.load(f)
    id2composition = {
        int(index): composition for index, composition in id2composition_str.items()
    }
    composition2id = {
        composition: index for index, composition in id2composition.items()
    }
    num_labels = len(id2composition)

    # Model loading
    # ----------------------------
    ic(f">> Loading model from '{model_path}'")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        id2label=id2composition,
        label2id=composition2id,
        problem_type="multi_label_classification",
    )
    model.to(device)

    # Composition inference
    # ----------------------------
    ic(f">> Initializing inference for {num_labels} labels")
    inference_pipe = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_all_scores=True,
    )

    # Validation set
    ic("> Running validation inference")
    # Generate predictions using pipeline
    predictions_val = [
        output
        for output in tqdm(
            inference_pipe(KeyDataset(dataset_val, "input")), total=len(dataset_val)
        )
    ]

    data_with_predictions_val = [
        {
            "index": j["index"],
            "input": j["input"],
            "pred_probabilities": [k["score"] for k in predictions_val[i]],
            "pred_best_composition": predictions_val[i][
                np.argmax([k["score"] for k in predictions_val[i]])
            ]["label"],
        }
        for i, j in enumerate(data_with_label_vectors_val)
    ]

    ic("> Running test inference")
    # Generate predictions using pipeline
    predictions_test = [
        output
        for output in tqdm(
            inference_pipe(KeyDataset(dataset_test, "input")), total=len(dataset_test)
        )
    ]
    data_with_predictions_test = [
        {
            "index": j["index"],
            "input": j["input"],
            "pred_probabilities": [k["score"] for k in predictions_test[i]],
            "pred_best_composition": predictions_test[i][
                np.argmax([k["score"] for k in predictions_test[i]])
            ]["label"],
        }
        for i, j in enumerate(data_with_label_vectors_test)
    ]

    # --------------------------
    # Saving all to file
    results_save_path_dir = f"outputs/composition-predictions/{model_path_basename}"
    makedirs(results_save_path_dir, exist_ok=True)

    # Export results file and id2component map to file
    pd.DataFrame(data=data_with_predictions_val).to_parquet(
        path.join(results_save_path_dir, f"{args.data}_val_results.parquet")
    )
    pd.DataFrame(data=data_with_predictions_test).to_parquet(
        path.join(results_save_path_dir, f"{args.data}_test_results.parquet")
    )
    with open(path.join(results_save_path_dir, "id2component_map.json"), "w") as f:
        json.dump(id2composition_str, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help=("Path to the model that will be used to predict compositions."),
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
