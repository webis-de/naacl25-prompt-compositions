import argparse
from os import path

import numpy as np
from icecream import ic
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from datasets import Dataset
from src.bias_detection.config.settings import MAX_SEQUENCE_LENGTH
from src.bias_detection.data_handler import DataHandler
from utils.general import get_custom_timestamp_string

ic.configureOutput(prefix=get_custom_timestamp_string)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_model",
    type=str,
    help="Path or name to the base encoder model that will be finetuned.",
    required=True,
)
parser.add_argument(
    "--cache_path",
    type=str,
    help=(
        "Path to the directory were datasets can be cached. This should be a location "
        "with fast I/O."
    ),
    default="/tmp",
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
    "--seed", type=int, help="Seed to be used for random initializations", default=42
)
parser.add_argument("--data", type=str, help="Dataset to train on.", required=True)
args = parser.parse_args()


MODEL_PATH = args.base_model
MODEL_NAME = path.basename(MODEL_PATH)
DATA_CACHE_PATH = path.join(args.cache_path)
SEED = args.seed

# Hyperparameters
BATCH_SIZE = 64
TRAIN_EPOCHS = 3

# Set seeds for reproducable results
set_seed(seed=args.seed)
# enable_full_determinism(seed=args.seed)  # significantly impacts training speed


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "macro_F1": precision_recall_fscore_support(
            labels, predictions, average="macro"
        )[2],
        "micro_F1": precision_recall_fscore_support(
            labels, predictions, average="micro"
        )[2],
        "binary_F1_postive": precision_recall_fscore_support(
            labels, predictions, average="binary", pos_label=1
        )[2],
        "binary_F1_negative": precision_recall_fscore_support(
            labels, predictions, average="binary", pos_label=0
        )[2],
    }


def preprocess_text(samples):
    return tokenizer(
        samples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
    )


ic("============================================================")
ic("===Loading data")
if args.data == "sbic":
    data_handler = DataHandler(datasets_to_load=["sbic"])

    train_df = data_handler.sbic_data["train"].rename(
        columns={"post": "text", "hasBiasedImplication": "label"}
    )[["text", "label"]]
    train_df = Dataset.from_pandas(train_df)

    dev_df = data_handler.sbic_data["dev"].rename(
        columns={"post": "text", "hasBiasedImplication": "label"}
    )[["text", "label"]]
    dev_df = Dataset.from_pandas(dev_df)
elif args.data == "stereoset":
    data_handler = DataHandler(datasets_to_load=["stereoset"])

    train_df = data_handler.stereoset_data["train"][
        ["text", "hasBiasedImplication"]
    ].rename(columns={"post": "text", "hasBiasedImplication": "label"})
    train_df = Dataset.from_pandas(train_df)

    dev_df = data_handler.stereoset_data["dev"][
        ["text", "hasBiasedImplication"]
    ].rename(columns={"post": "text", "hasBiasedImplication": "label"})
    dev_df = Dataset.from_pandas(dev_df)
elif args.data == "cobra_frames":
    data_handler = DataHandler(datasets_to_load=["cobra_frames"])

    train_df = data_handler.cobra_frames["train"][
        ["post", "hasBiasedImplication"]
    ].rename(columns={"post": "text", "hasBiasedImplication": "label"})
    train_df = Dataset.from_pandas(train_df)

    dev_df = data_handler.cobra_frames["dev"][["post", "hasBiasedImplication"]].rename(
        columns={"post": "text", "hasBiasedImplication": "label"}
    )
    dev_df = Dataset.from_pandas(dev_df)
elif args.data == "semeval":
    data_handler = DataHandler(datasets_to_load=["semeval"])

    train_df = data_handler.semeval_data["train"].rename(columns={"post": "text"})[
        ["text", "label"]
    ]
    train_df = Dataset.from_pandas(train_df)

    dev_df = data_handler.semeval_data["dev"].rename(columns={"post": "text"})[
        ["text", "label"]
    ]
    dev_df = Dataset.from_pandas(dev_df)
elif args.data == "esnli":
    data_handler = DataHandler(datasets_to_load=["esnli"])

    train_df = data_handler.esnli_data["train"].rename(columns={"post": "text"})[
        ["text", "label"]
    ]
    train_df = Dataset.from_pandas(train_df)

    dev_df = data_handler.esnli_data["dev"].rename(columns={"post": "text"})[
        ["text", "label"]
    ]
    dev_df = Dataset.from_pandas(dev_df)
elif args.data == "commonsense_qa":
    data_handler = DataHandler(datasets_to_load=["common_qa"])

    train_df = data_handler.common_qa["train"]
    train_df["qa_concat"] = train_df.apply(
        lambda row: f"[Q] {row['question']} [A] {row['answer']}", axis=1
    )
    train_df = train_df.rename(columns={"qa_concat": "text"})[["text", "label"]]
    train_df = Dataset.from_pandas(train_df)

    dev_df = data_handler.common_qa["dev"]
    dev_df["qa_concat"] = dev_df.apply(
        lambda row: f"[Q] {row['question']} [A] {row['answer']}", axis=1
    )
    dev_df = dev_df.rename(columns={"qa_concat": "text"})[["text", "label"]]
    dev_df = Dataset.from_pandas(dev_df)
else:
    raise ValueError("Dataset not implemented.")


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

ic("==Preparing train split")
train_dataset = train_df.map(
    preprocess_text,
    batched=True,
    cache_file_name=f"{DATA_CACHE_PATH}/ft-hf-train_{args.model_identifier}.arrow",
    num_proc=12,
)
ic("==Preparing val split")
val_dataset = dev_df.map(
    preprocess_text,
    batched=True,
    cache_file_name=f"{DATA_CACHE_PATH}/ft-hf-dev_{args.model_identifier}.arrow",
    num_proc=12,
)

ic("============================================================")
ic(f"===Loading model '{MODEL_NAME}'")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
)

ic("============================================================")
ic("===Starting model training")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
if args.model_identifier == "":
    model_path_basename = f"{MODEL_NAME}-seed{SEED}"
else:
    model_path_basename = f"{MODEL_NAME}-{args.model_identifier}-seed{SEED}"

ic(f"==Saving model-related files with basename {model_path_basename} in paths.")
training_args = TrainingArguments(
    logging_dir=f"./logs/{model_path_basename}",
    per_device_train_batch_size=BATCH_SIZE,
    do_eval=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    greater_is_better=True,
    num_train_epochs=TRAIN_EPOCHS,
    eval_steps=200,
    save_steps=500,
    logging_steps=10,
    output_dir=f"intermediate/{model_path_basename}-checkpoints",
)
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=val_dataset,
    train_dataset=train_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

ic("==Saving best model")
model_save_path = f"models/{model_path_basename}"
trainer.save_model(model_save_path)


ic("Done.")
