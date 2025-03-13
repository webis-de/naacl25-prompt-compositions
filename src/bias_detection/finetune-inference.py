import argparse
from os import path

from icecream import ic
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset

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
    "--model",
    type=str,
    help="Base model that was finetuned to retrieve the inference model. Mainly important for "
    "loading the correct tokenizer.",
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
parser.add_argument("--data", type=str, help="Dataset to train on.", required=True)
args = parser.parse_args()

MODEL_PATH = args.model
MODEL_NAME = path.basename(MODEL_PATH)
BASE_MODEL_PATH = args.base_model
BASE_MODEL_NAME = path.basename(BASE_MODEL_PATH)
DATA_CACHE_PATH = path.join(args.cache_path)

# Hyperparameters
BATCH_SIZE = 64


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

    test_df = data_handler.sbic_data["test"].rename(
        columns={"post": "text", "hasBiasedImplication": "label"}
    )
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])
elif args.data == "stereoset":
    data_handler = DataHandler(datasets_to_load=["stereoset"])

    test_df = data_handler.stereoset_data["test"][
        ["text", "hasBiasedImplication", "text_hash"]
    ].rename(columns={"hasBiasedImplication": "label", "text_hash": "md5_hash"})
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])
elif args.data == "cobra_frames":
    data_handler = DataHandler(datasets_to_load=["cobra_frames"])

    test_df = data_handler.cobra_frames["test"][
        ["post", "hasBiasedImplication", "text_hash"]
    ].rename(
        columns={
            "post": "text",
            "hasBiasedImplication": "label",
            "text_hash": "md5_hash",
        }
    )
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])
elif args.data == "semeval":
    data_handler = DataHandler(datasets_to_load=["semeval"])

    test_df = data_handler.semeval_data["test"].rename(columns={"post": "text"})
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])
elif args.data == "esnli":
    data_handler = DataHandler(datasets_to_load=["esnli"])

    test_df = data_handler.esnli_data["test"].rename(columns={"post": "text"})
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])
elif args.data == "commonsense_qa":
    data_handler = DataHandler(datasets_to_load=["common_qa"])

    test_df = data_handler.common_qa["test"]
    test_df["qa_concat"] = test_df.apply(
        lambda row: f"[Q] {row['question']} [A] {row['answer']}", axis=1
    )
    test_df = test_df.rename(columns={"qa_concat": "text"})
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])
else:
    raise ValueError("Dataset not implemented.")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

ic("==Preparing test split")
test_dataset = test_ds.map(
    preprocess_text,
    batched=True,
    cache_file_name=f"{DATA_CACHE_PATH}/ft-hf-test_{MODEL_NAME}.arrow",
    num_proc=12,
)

ic("============================================================")
ic(f"===Loading model '{MODEL_NAME}'")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
)


ic("============================================================")
ic("===Starting model inference")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Create sequence classification pipeline
inference_pipe = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    batch_size=BATCH_SIZE,
    max_length=MAX_SEQUENCE_LENGTH,
    truncation=True,
)

test_key_dataset = KeyDataset(test_dataset, "text")

# Generate predictions using pipeline
predictions = [
    output for output in tqdm(inference_pipe(test_key_dataset), total=len(test_dataset))
]

# Extract the label binary value from predicted string label
prediction_column_label = f"prediction_{MODEL_NAME}"
test_df.loc[:, prediction_column_label] = [
    int(pred["label"][-1]) for pred in predictions
]

ic("============================================================")
# Export prediction results to file
ic("===Writing results to file")
prediction_column_label = f"prediction_{MODEL_NAME}"
test_df[["md5_hash", prediction_column_label]].to_parquet(
    path.join("results", f"{args.data}-test_predictions-{MODEL_NAME}.parquet")
)


ic("Done.")
