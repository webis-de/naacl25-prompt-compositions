import json
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sentence_transformers import util
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from .config import (
    COBRA_NON_OFFENSIVE_HINTS,
    COBRA_OFFENSIVE_HINTS,
    COBRA_SPLIT_SIZE,
    COBRA_STATEMENT_TEMPLATE,
    DATASET_PATHS,
    EMBEDDING_MODEL_NAME,
    FILE_NAME_TEMPLATES,
    NONE_CATEGORY_NAME,
    NUM_SIMILAR_EXAMPLES,
    PATHS,
    RANDOM_SEED,
    SBIC_DATA_SPLITS,
    SBIC_RANDOM_SPLIT_SIZE,
    SENTENCE_TRANSFORMER_MODELS,
    Binary_bias_labels,
    SBIC_bias_categories,
)
from .utils import (
    ceil_to_next_even,
    clean_string,
    compute_sentence_embeddings,
    concat_sentences_stereoset,
    create_balanced_subset,
    get_similar_texts_as_examples,
    remove_duplicate_posts,
)


class DataHandler:
    def __init__(self, datasets_to_load: Union[List[str], str]) -> None:
        """Data handler class to load and preprocess datasets."""
        if isinstance(datasets_to_load, str):
            datasets_to_load = [datasets_to_load]

        self._paths = DATASET_PATHS
        self._cache_path = PATHS["intermediate"]
        self._emb_model_name = EMBEDDING_MODEL_NAME
        self._emb_model = SENTENCE_TRANSFORMER_MODELS[self._emb_model_name]

        self.sbic_data = None
        self.stereoset_data = None
        self.cobra_frames = None
        self.semeval_data = None
        self.esnli_data = None
        self.common_qa = None

        self._sbic_categories = [category.value for category in SBIC_bias_categories]
        self._sbic_labels = [label.value for label in Binary_bias_labels]
        self._sbic_md5_lookup = dict()
        self._sbic_property_lookup = dict()

        self._stereoset_md5_lookup = dict()
        self._stereoset_property_lookup = dict()

        self._cobra_md5_lookup = dict()
        self._cobra_property_lookup = dict()

        self._semeval_md5_lookup = dict()
        self._semeval_property_lookup = dict()

        self._esnli_md5_lookup = dict()
        self._esnli_property_lookup = dict()

        self._common_qa_md5_lookup = dict()
        self._common_qa_property_lookup = dict()

        # load all specified datasets
        for dataset in datasets_to_load:
            if dataset.lower() == "sbic":
                self.sbic_data = self.load_sbic_data()
            elif dataset.lower() == "stereoset":
                self.stereoset_data = self.load_stereoset_data()
            elif dataset.lower() == "cobra_frames":
                self.cobra_frames = self.load_cobra_frames_data()
            elif dataset.lower() == "semeval":
                self.semeval_data = self.load_semeval_data()
            elif dataset.lower() == "esnli":
                self.esnli_data = self.load_esnli_data()
            elif dataset.lower() == "common_qa":
                self.common_qa = self.load_common_qa_data()
            else:
                raise ValueError(f"Dataset {dataset} not found")

    def load_sbic_data(self) -> Dict[str, pd.DataFrame]:
        """
        Loads the Social Bias Inference Corpus (SBIC) from source, preprocesses
        it, and organizes it into a dictionary of dataframes split by train, test, and development
        sets. In addition it adds the new splits 'train_sub_split_random' and
        'train_sub_split_balanced'. Checks if the preprocessed data is already available as parquet
        files in the cache directory and loads them if available.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing dataframes for different splits:
            - "train": The training data.
            - "test": The testing data.
            - "dev": The development/validation data.
            - "train_sub_split_random": A random subset of the training data.
            - "train_sub_split_balanced": A balanced subset of the training data.
        """
        # check if data was already loaded in the data handler
        if self.sbic_data is not None:
            return self.sbic_data

        paths = [
            FILE_NAME_TEMPLATES["preprocessed_sbic"].format(split=s)
            for s in SBIC_DATA_SPLITS
        ]

        # if any of the preprocessed files are missing, preprocess the data again
        if not all(Path(path).is_file() for path in paths):
            # {split: df} for all splits
            sbic_data = self._load_and_preprocess_sbic_from_source()

            # add similarity based few-shot examples to each df as a column
            sbic_data_with_similar_examples = {
                split: self._create_sbic_few_shot_examples(df, split)
                for split, df in sbic_data.items()
            }

            for split, df in sbic_data_with_similar_examples.items():
                df.to_parquet(
                    FILE_NAME_TEMPLATES["preprocessed_sbic"].format(split=split),
                    engine="pyarrow",
                )

        # read the preprocessed data from the cache directory
        self.sbic_data = dict()
        for split in SBIC_DATA_SPLITS:
            df = pd.read_parquet(
                FILE_NAME_TEMPLATES["preprocessed_sbic"].format(split=split)
            )
            self.sbic_data[split] = df

            self._update_sbic_md5_lookup(df["post"].tolist(), df["md5_hash"].tolist())
            self._update_sbic_property_lookup(df)

        return self.sbic_data

    def load_sbic_data_as_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load the SBIC dataset with the posts, labels, categories and the similarity based examples.
        Also, create a hash lookup for the posts. The dataset is split into
        train, test, dev, train_sub_split_random, and train_sub_split_balanced.

        Returns:
            Dict[str, Dict[str, List[str]]]: A dictionary containing dictionaries for different
            splits:
            - "train": The training data.
            - "test": The testing data.
            - "dev": The development/validation data.
            - "train_sub_split_random": A random subset of the training data.
            - "train_sub_split_balanced": A balanced subset of the training data.
        """
        sbic_data = self.load_sbic_data()

        return {
            split: {
                "posts": sbic_data[split]["post"].tolist(),
                "labels": sbic_data[split]["hasBiasedImplication"].tolist(),
                "categories": sbic_data[split]["targetCategory"].tolist(),
                "similar_examples": sbic_data[split]["similar_texts_by_label"].tolist(),
                "hash_lookup": self._sbic_md5_lookup,
                "offensive": sbic_data[split]["offensiveYN"].tolist(),
                "group": sbic_data[split]["targetMinority"].tolist(),
                "implied_statement": sbic_data[split]["targetStereotype"].tolist(),
                "property_lookup": self._sbic_property_lookup,
            }
            for split in [
                "test",
                "train",
                "dev",
                "train_sub_split_random",
                "train_sub_split_balanced",
            ]
        }

    def _load_and_preprocess_sbic_from_source(self) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess the SBIC data from the original dataset.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing dataframes for different splits:
            - "train": The training data.
            - "test": The testing data.
            - "dev": The development/validation data.
            - "train_sub_split_random": A random subset of the training data.
            - "train_sub_split_balanced": A balanced subset of the training data.
        """

        # create the sub splits for the training data
        df_train = self._process_sbic_data(f"{self._paths['sbic']}/SBIC.v2.agg.trn.csv")
        df_train_sub_split_balanced = create_balanced_subset(
            df_train, subset_size=SBIC_RANDOM_SPLIT_SIZE, seed=RANDOM_SEED[0]
        )
        df_train_sub_split_random = df_train.sample(
            n=SBIC_RANDOM_SPLIT_SIZE, random_state=RANDOM_SEED[0]
        )

        return {
            "train": df_train,
            "test": self._process_sbic_data(
                f"{self._paths['sbic']}/SBIC.v2.agg.tst.csv"
            ),
            "dev": self._process_sbic_data(
                f"{self._paths['sbic']}/SBIC.v2.agg.dev.csv"
            ),
            "train_sub_split_random": df_train_sub_split_random,
            "train_sub_split_balanced": df_train_sub_split_balanced,
        }

    def _process_sbic_data(self, file_path: str) -> pd.DataFrame:
        """
        Processes a given SBIC CSV file by:
        - Reversing the encoding of the 'hasBiasedImplication' column to make the bias
            labeling more intuitive (0 for no bias, 1 for bias).
        - Loading columns that contain lists of strings (e.g., 'targetMinority',
            'targetCategory', and 'targetStereotype') as actual lists instead of
            strings.
        - Cleaning the text in the 'post' column to remove or standardize certain
            characters or patterns.
        - Computing or loading the most similar texts for each sample in the dataset.

        Args:
            file_path (str): Path to the CSV file containing the SBIC split.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        df = pd.read_csv(file_path)

        df = df.drop("Unnamed: 0", axis=1)

        # the 'hasBiasedImplication' column contains the bias label encoded as 0 or 1
        # by default 0 represents bias and 1 represents no bias
        # since this is counterintuitive, we reverse the values
        # 0 -> no bias, 1 -> bias
        df["hasBiasedImplication"] = (
            df["hasBiasedImplication"].astype(int).apply(lambda x: 1 if x == 0 else 0)
        )

        df["targetMinority"] = df["targetMinority"].apply(json.loads)
        df["targetCategory"] = df["targetCategory"].apply(json.loads)
        df["targetStereotype"] = df["targetStereotype"].apply(json.loads)

        df["post"] = df["post"].apply(lambda x: clean_string(x, replace_link=True))

        # there are duplicate posts in the dataset
        df = remove_duplicate_posts(df)

        df["md5_hash"] = df["post"].apply(lambda x: md5(x.encode()).hexdigest())

        return df

    def _update_sbic_md5_lookup(
        self, clean_posts: List[str], md5_hashes: List[str]
    ) -> None:
        """
        Update the SBIC MD5 lookup table with clean posts and their corresponding MD5 hashes.

        Args:
            clean_posts (List[str]): A list of cleaned post texts.
            md5_hashes (List[str]): A list of corresponding MD5 hashes.
        """
        tmp_lookup = dict(zip(md5_hashes, clean_posts))

        if not self._sbic_md5_lookup:
            self._sbic_md5_lookup = tmp_lookup

        self._sbic_md5_lookup.update(tmp_lookup)

    def _update_sbic_property_lookup(self, df: pd.DataFrame) -> None:
        """
        Update the SBIC property lookup table with the offensive, group, and implied statement.

        Args:
            df (pd.DataFrame): SBIC data.

        Raises:
            ValueError: If the properties of the SBIC do not have the same length.
        """
        offensive = df["offensiveYN"].tolist()
        group = df["targetMinority"].tolist()
        implied_statement = df["targetStereotype"].tolist()
        hashes = df["md5_hash"].tolist()

        if not (len(offensive) == len(group) == len(implied_statement) == len(hashes)):
            raise ValueError("All properties of the SBIC must have the same length.")

        tmp_lookup = {
            hashes[idx]: {
                "offensive": offensive[idx],
                "group": group[idx],
                "implied_statement": implied_statement[idx],
            }
            for idx in range(len(hashes))
        }

        if not self._sbic_property_lookup:
            self._sbic_property_lookup = tmp_lookup

        self._sbic_property_lookup.update(tmp_lookup)

    def _create_sbic_few_shot_examples(
        self, df: pd.DataFrame, split: str
    ) -> pd.DataFrame:
        """
        Create few-shot examples for the SBIC dataset.

        Args:
            df (pd.DataFrame): The input dataframe containing SBIC data.
            split (str): The data split ("train", "test", or "dev").

        Returns:
            pd.DataFrame: The dataframe with added few-shot examples.
        """
        example_hashes = get_similar_texts_as_examples(
            df["post"].tolist(),
            df["hasBiasedImplication"].tolist(),
            df["md5_hash"].tolist(),
            emb_model_path=self._emb_model,
            n=NUM_SIMILAR_EXAMPLES,
        )

        examples_df = pd.DataFrame(
            {
                "md5_hash": df["md5_hash"].tolist(),
                "similar_texts_by_label": example_hashes,
            }
        )

        return pd.merge(df, examples_df, on="md5_hash")

    def load_semeval_data(self) -> Dict[str, pd.DataFrame]:
        """
        Loads the SemEval dataset from preprocessed parquet files if available. If the preprocessed
        files are missing, preprocesses the data from the source and saves it.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing dataframes for different splits:
            - "train": The training data.
            - "dev": The development/validation data.
            - "test": The testing data.
        """
        # check if data was already loaded in the data handler
        if self.semeval_data is not None:
            return self.semeval_data

        paths = [
            FILE_NAME_TEMPLATES["preprocessed_semeval"].format(split=s)
            for s in ["train", "dev", "test"]
        ]

        # if any of the preprocessed files are missing, preprocess the data again
        if not all(Path(path).is_file() for path in paths):
            # {split: df} for all splits
            semeval_data = self._load_and_preprocess_semeval_from_source()

            # add similarity based few-shot examples to each df as a column
            semeval_data_with_similar_examples = {
                split: self._create_semeval_few_shot_examples(df, split)
                for split, df in semeval_data.items()
            }

            for split, df in semeval_data_with_similar_examples.items():
                df.to_parquet(
                    FILE_NAME_TEMPLATES["preprocessed_semeval"].format(split=split),
                    engine="pyarrow",
                )

        # read the preprocessed data from the cache directory
        self.semeval_data = dict()
        for split in ["train", "test", "dev"]:
            df = pd.read_parquet(
                FILE_NAME_TEMPLATES["preprocessed_semeval"].format(split=split)
            )
            self.semeval_data[split] = df

            self._update_semeval_md5_lookup(
                df["post"].tolist(), df["md5_hash"].tolist()
            )
            self._update_semeval_property_lookup(df)

        return self.semeval_data

    def load_semeval_data_as_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Loads the SemEval dataset and returns it in a dictionary format. The dictionary contains
        posts, labels, categories, aspect terms, polarities, and similarity-based examples, along
        with lookup tables.

        Returns:
            Dict[str, Dict[str, List[str]]]: A dictionary with the following keys for each split:
            - "posts": List of post texts.
            - "labels": List of sentiment labels.
            - "categories": List of source categories.
            - "similar_examples": List of similar examples by label.
            - "aspect_terms": List of aspect terms.
            - "polarities": List of sentiment polarities.
            - "hash_lookup": Dictionary for MD5 hash lookup.
            - "property_lookup": Dictionary for property lookup by hash.
        """
        semeval_data = self.load_semeval_data()

        return {
            split: {
                "posts": semeval_data[split]["post"].tolist(),
                "labels": semeval_data[split]["label"].tolist(),
                "categories": semeval_data[split]["source"].tolist(),
                "similar_examples": semeval_data[split][
                    "similar_texts_by_label"
                ].tolist(),
                "aspect_terms": semeval_data[split]["aspect_terms"].tolist(),
                "polarities": semeval_data[split]["polarities"].tolist(),
                "hash_lookup": self._semeval_md5_lookup,
                "property_lookup": self._semeval_property_lookup,
            }
            for split in [
                "test",
                "train",
                "dev",
            ]
        }

    def _load_and_preprocess_semeval_from_source(self) -> Dict[str, pd.DataFrame]:
        """
        Loads and preprocesses the SemEval dataset from the source, including renaming and
        relabeling, filtering out neutral examples, and creating splits.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing dataframes for different splits:
            - "train": The training data.
            - "dev": The development/validation data.
            - "test": The testing data.
        """
        df = pd.read_parquet(f"{DATASET_PATHS['semeval']}/semeval-2014-absa.parquet")

        df = df[df["label"] != "neutral"]

        df["sentiment"] = df["label"]
        df["label"] = df["label"].apply(lambda x: 1 if x == "positive" else 0)
        df = df.rename(columns={"md5hash": "md5_hash"})

        return {
            "train": df[df["split"] == "train"],
            "test": df[df["split"] == "test"],
            "dev": df[df["split"] == "dev"],
        }

    def _create_semeval_few_shot_examples(
        self, df: pd.DataFrame, split: str
    ) -> pd.DataFrame:
        """
        Creates few-shot examples for the SemEval dataset by finding the most similar texts
        for each example.

        Args:
            df (pd.DataFrame): The input dataframe containing SemEval data.
            split (str): The data split ("train", "test", or "dev").

        Returns:
            pd.DataFrame: The dataframe with added few-shot examples.
        """
        example_hashes = get_similar_texts_as_examples(
            df["post"].tolist(),
            df["label"].tolist(),
            df["md5_hash"].tolist(),
            emb_model_path=self._emb_model,
            n=NUM_SIMILAR_EXAMPLES,
        )

        examples_df = pd.DataFrame(
            {
                "md5_hash": df["md5_hash"].tolist(),
                "similar_texts_by_label": example_hashes,
            }
        )

        return pd.merge(df, examples_df, on="md5_hash")

    def _update_semeval_md5_lookup(
        self, texts: List[str], md5_hashes: List[str]
    ) -> None:
        """
        Updates the SemEval MD5 lookup table with texts and their corresponding MD5 hashes.

        Args:
            texts (List[str]): A list of texts.
            md5_hashes (List[str]): A list of corresponding MD5 hashes.
        """
        tmp_lookup = dict(zip(md5_hashes, texts))

        if not self._semeval_md5_lookup:
            self._semeval_md5_lookup = tmp_lookup

        self._semeval_md5_lookup.update(tmp_lookup)

    def _update_semeval_property_lookup(self, df: pd.DataFrame) -> None:
        """
        Updates the SemEval property lookup table with aspect terms, polarities, and categories.

        Args:
            df (pd.DataFrame): SemEval data in DataFrame format.
        """
        aspect_terms = df["aspect_terms"].tolist()
        polarities = df["polarities"].tolist()
        source = df["source"].tolist()
        hashes = df["md5_hash"].tolist()

        tmp_lookup = {
            hashes[idx]: {
                "aspect_terms": aspect_terms[idx],
                "polarities": polarities[idx],
                "category": source[idx],
            }
            for idx in range(len(hashes))
        }

        if not self._semeval_property_lookup:
            self._semeval_property_lookup = tmp_lookup

        self._semeval_property_lookup.update(tmp_lookup)

    def load_esnli_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load the e-SNLI dataset from preprocessed files if available. If the preprocessed files
        are missing, load and preprocess the data from the source.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary with keys "train", "dev", and "test" containing
            the respective preprocessed e-SNLI splits.
        """
        # check if data was already loaded in the data handler
        if self.esnli_data is not None:
            return self.esnli_data

        paths = [
            FILE_NAME_TEMPLATES["preprocessed_esnli"].format(split=s)
            for s in ["train", "dev", "test"]
        ]

        # if any of the preprocessed files are missing, preprocess the data again
        if not all(Path(path).is_file() for path in paths):
            # {split: df} for all splits
            esnli_data = self._load_and_preprocess_esnli_from_source()

            # add similarity based few-shot examples to each df as a column
            esnli_data_with_similar_examples = {
                split: self._create_esnli_few_shot_examples(df, split)
                for split, df in esnli_data.items()
            }

            for split, df in esnli_data_with_similar_examples.items():
                df.to_parquet(
                    FILE_NAME_TEMPLATES["preprocessed_esnli"].format(split=split),
                    engine="pyarrow",
                )

        # read the preprocessed data from the cache directory
        self.esnli_data = dict()
        for split in ["train", "test", "dev"]:
            df = pd.read_parquet(
                FILE_NAME_TEMPLATES["preprocessed_esnli"].format(split=split)
            )
            self.esnli_data[split] = df

            self._update_esnli_md5_lookup(df["post"].tolist(), df["md5_hash"].tolist())
            self._update_esnli_property_lookup(df)

        return self.esnli_data

    def load_esnli_data_as_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load the e-SNLI dataset and return it in dictionary format with posts, labels,
        explanations, and similar examples.

        Returns:
            Dict[str, Dict[str, List[str]]]: A dictionary with keys "train", "dev", and "test"
            each containing:
            - "posts": List of sentence pairs (premise and hypothesis).
            - "labels": List of labels indicating entailment, contradiction, or neutral.
            - "explanations": List of human-written explanations.
            - "similar_examples": List of similar examples based on sentence embeddings.
        """
        esnli_data = self.load_esnli_data()

        return {
            split: {
                "posts": esnli_data[split]["post"].tolist(),
                "labels": esnli_data[split]["label"].tolist(),
                "categories": esnli_data[split]["category"].tolist(),
                "similar_examples": esnli_data[split][
                    "similar_texts_by_label"
                ].tolist(),
                "explanations": esnli_data[split]["explanation"].tolist(),
                "hash_lookup": self._esnli_md5_lookup,
                "property_lookup": self._esnli_property_lookup,
            }
            for split in [
                "test",
                "train",
                "dev",
            ]
        }

    def _load_and_preprocess_esnli_from_source(self) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess the e-SNLI dataset from the original source, generating concatenated
        premise and hypothesis, and preparing data splits.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary with preprocessed splits for "train", "dev",
            and "test".
        """
        result = {}
        for split in ["train", "dev", "test"]:
            df = pd.read_parquet(
                f"{DATASET_PATHS['esnli']}/filtered_esnli_{split}.parquet"
            )

            df["post"] = (
                "Premise: " + df["premise"] + " Hypothesis: " + df["hypothesis"]
            )
            df["category"] = "general"

            df["md5_hash"] = df["post"].apply(lambda x: md5(x.encode()).hexdigest())

            result[split] = df

        return result

    def _create_esnli_few_shot_examples(
        self, df: pd.DataFrame, split: str
    ) -> pd.DataFrame:
        """
        Create few-shot examples for the e-SNLI dataset by finding the most similar sentence pairs.

        Args:
            df (pd.DataFrame): The input dataframe containing e-SNLI data.
            split (str): The data split ("train", "test", or "dev").

        Returns:
            pd.DataFrame: The dataframe with added few-shot examples.
        """
        example_hashes = get_similar_texts_as_examples(
            df["post"].tolist(),
            df["label"].tolist(),
            df["md5_hash"].tolist(),
            emb_model_path=self._emb_model,
            n=NUM_SIMILAR_EXAMPLES,
        )

        examples_df = pd.DataFrame(
            {
                "md5_hash": df["md5_hash"].tolist(),
                "similar_texts_by_label": example_hashes,
            }
        )

        return pd.merge(df, examples_df, on="md5_hash")

    def _update_esnli_md5_lookup(self, texts: List[str], md5_hashes: List[str]) -> None:
        """
        Update the e-SNLI MD5 lookup table with sentence pairs and their corresponding MD5 hashes.

        Args:
            texts (List[str]): List of concatenated premise and hypothesis pairs.
            md5_hashes (List[str]): List of corresponding MD5 hashes.
        """
        tmp_lookup = dict(zip(md5_hashes, texts))

        if not self._esnli_md5_lookup:
            self._esnli_md5_lookup = tmp_lookup

        self._esnli_md5_lookup.update(tmp_lookup)

    def _update_esnli_property_lookup(self, df: pd.DataFrame) -> None:
        """
        Update the e-SNLI property lookup table with explanations and their corresponding MD5 hashes.

        Args:
            df (pd.DataFrame): The e-SNLI data containing explanations.
        """
        explanation = df["explanation"].tolist()
        hashes = df["md5_hash"].tolist()

        tmp_lookup = {
            hashes[idx]: {
                "explanation": explanation[idx],
                "category": "general",
            }
            for idx in range(len(hashes))
        }

        if not self._esnli_property_lookup:
            self._esnli_property_lookup = tmp_lookup

        self._esnli_property_lookup.update(tmp_lookup)

    def load_common_qa_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load the Common QA dataset from preprocessed files if available. If the preprocessed files
        are missing, preprocess the data from the source.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing preprocessed "train", "dev", and
            "test" splits for the Common QA dataset.
        """
        # check if data was already loaded in the data handler
        if self.common_qa is not None:
            return self.common_qa

        paths = [
            FILE_NAME_TEMPLATES["preprocessed_common_qa"].format(split=s)
            for s in ["train", "dev", "test"]
        ]

        # if any of the preprocessed files are missing, preprocess the data again
        if not all(Path(path).is_file() for path in paths):
            # {split: df} for all splits
            common_qa = self._load_and_preprocess_common_qa_from_source()

            # add similarity based few-shot examples to each df as a column
            common_qa_data_with_similar_examples = {
                split: self._create_esnli_few_shot_examples(df, split)
                for split, df in common_qa.items()
            }

            for split, df in common_qa_data_with_similar_examples.items():
                df.to_parquet(
                    FILE_NAME_TEMPLATES["preprocessed_common_qa"].format(split=split),
                    engine="pyarrow",
                )

        # read the preprocessed data from the cache directory
        self.common_qa = dict()
        for split in ["train", "test", "dev"]:
            df = pd.read_parquet(
                FILE_NAME_TEMPLATES["preprocessed_common_qa"].format(split=split)
            )
            self.common_qa[split] = df

            self._update_common_qa_md5_lookup(
                df["post"].tolist(), df["md5_hash"].tolist()
            )
            self._update_common_qa_property_lookup(df)

        return self.common_qa

    def load_common_qa_data_as_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load the Common QA dataset and return it in dictionary format. The dictionary contains posts,
        labels, categories, concepts, distractors, and similarity-based examples, along with lookup
        tables for hashes and properties.

        Returns:
            Dict[str, Dict[str, List[str]]]: A dictionary where each key (train, test, dev) contains:
            - "posts": List of post texts.
            - "labels": List of corresponding labels.
            - "categories": List of source categories.
            - "similar_examples": List of similar texts by label.
            - "concept": List of concepts associated with each post.
            - "distractor": List of distractor texts.
            - "hash_lookup": Dictionary for MD5 hash lookup.
            - "property_lookup": Dictionary for property lookup by hash.
        """
        common_qa_data = self.load_common_qa_data()

        return {
            split: {
                "posts": common_qa_data[split]["post"].tolist(),
                "labels": common_qa_data[split]["label"].tolist(),
                "categories": common_qa_data[split]["category"].tolist(),
                "similar_examples": common_qa_data[split][
                    "similar_texts_by_label"
                ].tolist(),
                "concept": common_qa_data[split]["concept"].tolist(),
                "distractor": common_qa_data[split]["distractor"].tolist(),
                "hash_lookup": self._common_qa_md5_lookup,
                "property_lookup": self._common_qa_property_lookup,
            }
            for split in [
                "test",
                "train",
                "dev",
            ]
        }

    def _load_and_preprocess_common_qa_from_source(self) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess the Common QA dataset from the original source, generating train, dev,
        and test splits.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing preprocessed data for "train", "dev",
            and "test".
        """
        # load raw data from file
        df_train_test = pd.read_parquet(
            f"{DATASET_PATHS['common_qa']}/filtered_commonsense_qa_train_split.parquet"
        )
        df_dev = pd.read_parquet(
            f"{DATASET_PATHS['common_qa']}/filtered_commonsense_qa_dev_split.parquet"
        )

        # create test, train, and dev splits
        df_label_0 = df_train_test[df_train_test["label"] == 0]
        df_label_1 = df_train_test[df_train_test["label"] == 1]
        df_train = pd.concat([df_label_0.head(2500), df_label_1.head(2500)])
        df_test = pd.concat([df_label_0.iloc[2500:3000], df_label_1.iloc[2500:3000]])

        # format the dataframes to have the same columns as the other datasets
        df_train["post"] = (
            df_train["question"] + "\n"
            "Is '"
            + df_train["answer"]
            + "' the correct answer for the previous question?"
        )
        df_test["post"] = (
            df_test["question"] + "\n"
            "Is '"
            + df_test["answer"]
            + "' the correct answer for the previous question?"
        )
        df_dev["post"] = (
            df_dev["question"] + "\n"
            "Is '"
            + df_dev["answer"]
            + "' the correct answer for the previous question?"
        )
        df_train["category"] = "general"
        df_test["category"] = "general"
        df_dev["category"] = "general"

        df_train["md5_hash"] = df_train["post"].apply(
            lambda x: md5(x.encode()).hexdigest()
        )
        df_test["md5_hash"] = df_test["post"].apply(
            lambda x: md5(x.encode()).hexdigest()
        )
        df_dev["md5_hash"] = df_dev["post"].apply(lambda x: md5(x.encode()).hexdigest())

        return {
            "train": df_train,
            "test": df_test,
            "dev": df_dev,
        }

    def _create_common_qa_few_shot_examples(
        self, df: pd.DataFrame, split: str
    ) -> pd.DataFrame:
        """
        Create few-shot examples for the Common QA dataset by finding the most similar texts.

        Args:
            df (pd.DataFrame): The input dataframe containing Common QA data.
            split (str): The data split ("train", "test", or "dev").

        Returns:
            pd.DataFrame: The dataframe with an additional column 'similar_texts_by_label' that contains
            few-shot examples based on sentence similarity.
        """
        example_hashes = get_similar_texts_as_examples(
            df["post"].tolist(),
            df["label"].tolist(),
            df["md5_hash"].tolist(),
            emb_model_path=self._emb_model,
            n=NUM_SIMILAR_EXAMPLES,
        )

        examples_df = pd.DataFrame(
            {
                "md5_hash": df["md5_hash"].tolist(),
                "similar_texts_by_label": example_hashes,
            }
        )

        return pd.merge(df, examples_df, on="md5_hash")

    def _update_common_qa_md5_lookup(
        self, texts: List[str], md5_hashes: List[str]
    ) -> None:
        """
        Update the Common QA MD5 lookup table with texts and their corresponding MD5 hashes.

        Args:
            texts (List[str]): A list of post texts.
            md5_hashes (List[str]): A list of corresponding MD5 hashes for the posts.
        """
        tmp_lookup = dict(zip(md5_hashes, texts))

        if not self._common_qa_md5_lookup:
            self._common_qa_md5_lookup = tmp_lookup

        self._common_qa_md5_lookup.update(tmp_lookup)

    def _update_common_qa_property_lookup(self, df: pd.DataFrame) -> None:
        """
        Update the Common QA property lookup table with concepts and distractors.

        Args:
            df (pd.DataFrame): The dataframe containing Common QA data with 'concept' and
            'distractor' columns.
        """
        concept = df["concept"].tolist()
        distractor = df["distractor"].tolist()
        hashes = df["md5_hash"].tolist()

        tmp_lookup = {
            hashes[idx]: {
                "distractor": distractor[idx],
                "concept": concept[idx],
                "category": "general",
            }
            for idx in range(len(hashes))
        }

        if not self._common_qa_property_lookup:
            self._common_qa_property_lookup = tmp_lookup

        self._common_qa_property_lookup.update(tmp_lookup)

    def _load_and_preprocess_stereoset_from_source(self) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess the StereoSet dataset from the original source.

        - split sentences and gold labels in new sentences
        - add the context to the sentences
        - add md5 hash for each sentence
        - add category and label for each sentence
        - add similar_texts_by_label for each sentence
        - create balanced train, dev and test splits

        Returns:
            Dict[str, pd.DataFrame]: StereoSet data in a DataFrame format.
        """
        path = f"{DATASET_PATHS['stereoset']}/intersentence/validation-00000-of-00001.parquet"
        source_df = pd.read_parquet(path)

        id2category = {0: "anti-stereotype", 1: "stereotype", 2: "unrelated"}

        new_df_struct = {
            "id": [],
            "text_hash": [],
            "bias_type": [],
            "target": [],
            "text": [],
            "label_text": [],
            "label": [],
            "hasBiasedImplication": [],
        }

        for idx, row in source_df.iterrows():
            sentences = list(row["sentences"]["sentence"])
            gold_labels = list(row["sentences"]["gold_label"])

            ordered_sentences = concat_sentences_stereoset(
                sentences, gold_labels, row["context"]
            )

            for idx, s in enumerate(ordered_sentences):
                new_df_struct["id"].append(row["id"])
                new_df_struct["text_hash"].append(md5(s.encode()).hexdigest())
                new_df_struct["bias_type"].append(
                    row["bias_type"] if idx == 1 else NONE_CATEGORY_NAME
                )
                new_df_struct["target"].append(row["target"])
                new_df_struct["text"].append(s)
                new_df_struct["label_text"].append(id2category[idx])
                new_df_struct["label"].append(idx)
                new_df_struct["hasBiasedImplication"].append(1 if idx == 1 else 0)

        new_df_struct["similar_texts_by_label"] = get_similar_texts_as_examples(
            new_df_struct["text"],
            new_df_struct["hasBiasedImplication"],
            new_df_struct["text_hash"],
            emb_model_path=self._emb_model,
            n=NUM_SIMILAR_EXAMPLES,
        )

        # split the data 80/10/10
        data = pd.DataFrame(new_df_struct)
        train_dev, test = train_test_split(
            data, test_size=0.1, stratify=data["bias_type"], random_state=RANDOM_SEED[0]
        )
        train, dev = train_test_split(
            train_dev,
            test_size=0.1111,
            stratify=train_dev["bias_type"],
            random_state=RANDOM_SEED[0],
        )

        return {
            "train": train,
            "dev": dev,
            "test": test,
        }

    def load_stereoset_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load the StereoSet dataset from the cache directory if available. If the cache file is not
        available, preprocess the data and save it to the cache.

        Returns:
            Dict[str, pd.DataFrame]: StereoSet data in a DataFrame format.
        """
        if self.stereoset_data is not None:
            return self.stereoset_data

        data_splits = ["train", "dev", "test"]

        paths = [
            FILE_NAME_TEMPLATES["preprocessed_stereoset"].format(split=s)
            for s in data_splits
        ]

        # if any of the preprocessed files are missing, preprocess the data again
        if not all(Path(path).is_file() for path in paths):
            stereoset_data = self._load_and_preprocess_stereoset_from_source()

            for split, data in stereoset_data.items():
                path_cache = FILE_NAME_TEMPLATES["preprocessed_stereoset"].format(
                    split=split
                )
                data.to_parquet(path_cache, engine="pyarrow")

        self.stereoset_data = dict()
        for split in data_splits:
            df = pd.read_parquet(
                FILE_NAME_TEMPLATES["preprocessed_stereoset"].format(split=split)
            )
            self.stereoset_data[split] = df

            self._update_stereoset_md5_lookup(
                df["text"].tolist(), df["text_hash"].tolist()
            )
            self._update_stereoset_property_lookup(
                df["target"].tolist(), df["text_hash"].tolist()
            )

        return self.stereoset_data

    def load_stereoset_data_as_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load the StereoSet dataset with the posts, labels, categories and the similarity based
        examples. Also, create a hash lookup for the posts.

        Returns:
            Dict[str, Dict[str, List[str]]]: A dictionary with the following
            keys: "posts", "labels", "categories", "similar_examples", "hash_lookup".
        """
        stereoset_data = self.load_stereoset_data()

        return {
            split: {
                "posts": stereoset_data[split]["text"].tolist(),
                "labels": stereoset_data[split]["hasBiasedImplication"].tolist(),
                "categories": stereoset_data[split]["bias_type"].tolist(),
                "similar_examples": stereoset_data[split][
                    "similar_texts_by_label"
                ].tolist(),
                "hash_lookup": self._stereoset_md5_lookup,
                "property_lookup": self._stereoset_property_lookup,
                "targets": stereoset_data[split]["target"].tolist(),
            }
            for split in ["train", "dev", "test"]
        }

    def _update_stereoset_md5_lookup(
        self, texts: List[str], md5_hashes: List[str]
    ) -> None:
        """
        Update the Stereoset MD5 lookup table with texts and their corresponding MD5 hashes.

        Args:
            texts (List[str]): A list of texts.
            md5_hashes (List[str]): A list of corresponding MD5 hashes.
        """
        tmp_lookup = dict(zip(md5_hashes, texts))

        if not self._stereoset_md5_lookup:
            self._stereoset_md5_lookup = tmp_lookup

        self._stereoset_md5_lookup.update(tmp_lookup)

    def _update_stereoset_property_lookup(
        self, targets: List[str], md5_hashes: List[str]
    ) -> None:
        """
        Updates the internal stereoset property lookup dictionary with new targets and their
        corresponding MD5 hashes.

        Args:
            targets (List[str]): List of target strings to be added to the lookup.
            md5_hashes (List[str]): List of MD5 hash strings corresponding to the targets.
        """
        tmp_lookup = dict(zip(md5_hashes, targets))

        if not self._stereoset_property_lookup:
            self._stereoset_property_lookup = tmp_lookup

        self._stereoset_property_lookup.update(tmp_lookup)

    def load_cobra_frames_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load the Cobra Frames dataset from the cache directory if available.
        If the cache file is not available, preprocess the data and save it to the cache.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing dataframes for different splits:
            - "train": The training data.
            - "dev": The development/validation data.
            - "test": The testing data.
        """
        if self.cobra_frames is not None:
            return self.cobra_frames

        data_splits = ["train", "dev", "test"]

        paths = [
            FILE_NAME_TEMPLATES["preprocessed_cobra"].format(split=s)
            for s in data_splits
        ]

        # if any of the preprocessed files are missing, preprocess the data again
        if not all(Path(path).is_file() for path in paths):
            cobra_data = self._load_and_preprocess_cobra_frames_from_source()

            for split, data in cobra_data.items():
                path_cache = FILE_NAME_TEMPLATES["preprocessed_cobra"].format(
                    split=split
                )
                data.to_parquet(path_cache, engine="pyarrow")

        self.cobra_frames = dict()
        for split in data_splits:
            df = pd.read_parquet(
                FILE_NAME_TEMPLATES["preprocessed_cobra"].format(split=split)
            )
            self.cobra_frames[split] = df

            self._update_cobra_md5_lookup(df["post"].tolist(), df["text_hash"].tolist())
            self._update_cobra_property_lookup(df)

        return self.cobra_frames

    def load_cobra_frames_data_as_dict(self) -> Dict[str, List[str]]:
        """
        Load the Cobra Frames dataset and return it in a dictionary format.
        The dictionary contains posts, labels, categories, similar examples, and lookup tables.

        Returns:
            Dict[str, List[str]]: A dictionary containing dictionaries for different splits:
            - "train": The training data.
            - "dev": The development/validation data.
            - "test": The testing data.
        """
        cobra_data = self.load_cobra_frames_data()

        return {
            split: {
                "posts": cobra_data[split]["post"].tolist(),
                "labels": cobra_data[split]["hasBiasedImplication"].tolist(),
                "categories": cobra_data[split]["group"].tolist(),
                "similar_examples": cobra_data[split][
                    "similar_texts_by_label"
                ].tolist(),
                "hash_lookup": self._cobra_md5_lookup,
                "intents": cobra_data[split]["intent"].tolist(),
                "implications": cobra_data[split]["implication"].tolist(),
                "target_groups": cobra_data[split]["targetGroup"].tolist(),
                "property_lookup": self._cobra_property_lookup,
            }
            for split in ["train", "dev", "test"]
        }

    def _update_cobra_md5_lookup(self, texts: List[str], md5_hashes: List[str]) -> None:
        """
        Update the Cobra MD5 lookup table with texts and their corresponding MD5 hashes.

        Args:
            texts (List[str]): A list of texts.
            md5_hashes (List[str]): A list of corresponding MD5 hashes.
        """
        tmp_lookup = dict(zip(md5_hashes, texts))

        if not self._cobra_md5_lookup:
            self._cobra_md5_lookup = tmp_lookup

        self._cobra_md5_lookup.update(tmp_lookup)

    def _update_cobra_property_lookup(self, df: pd.DataFrame) -> None:
        """
        Update the Cobra property lookup table with intents, target groups, and implications.

        Args:
            df (pd.DataFrame): Cobra data.

        Raises:
            ValueError: If the properties of the Cobra dataset do not have the same length.
        """
        intent = df["intent"].tolist()
        target_group = df["targetGroup"].tolist()
        implication = df["implication"].tolist()
        hashes = df["text_hash"].tolist()

        if not (len(intent) == len(target_group) == len(implication) == len(hashes)):
            raise ValueError(
                "All properties of the Cobra dataset must have the same length."
            )

        tmp_lookup = {
            hashes[idx]: {
                "intent": intent[idx],
                "target_group": target_group[idx],
                "implication": implication[idx],
            }
            for idx in range(len(hashes))
        }

        if not self._cobra_property_lookup:
            self._cobra_property_lookup = tmp_lookup

        self._cobra_property_lookup.update(tmp_lookup)

    def _load_and_preprocess_cobra_frames_from_source(self) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess the Cobra Frames data from the original dataset.

        - Load data from the source files.
        - Create binary bias/offensiveness labels.
        - Create group column for the test split.
        - Remove groups with too few examples from the dataset.
        - Create train and dev splits based on the group and bias label.
        - Build statements based on a template and compute corresponding MD5 hash.
        - Find similar statements based on sentence embeddings for few-shot prompting.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing dataframes for different splits:
            - "train": The training data.
            - "dev": The development/validation data.
            - "test": The testing data.
        """
        # load the data from file
        df_test = pd.read_csv(
            f"{DATASET_PATHS['cobraframes']}/advContexts_explanations.csv"
        )
        df_full = pd.read_csv(
            f"{DATASET_PATHS['cobraframes']}/toxigen_explanations.csv"
        )

        df_test = df_test.dropna()
        df_full = df_full.dropna()

        # create binary bias/offensiveness label by checking for specific words in the offensiveness
        # column
        df_full = self._get_cobra_bias_labels(df_full)
        df_test = self._get_cobra_bias_labels(df_test)

        # create group column for the test split by computing the most similar group based on the
        # sentence embeddings of the statement + original target group column and the group names
        df_test = self._create_cobra_group_column(
            df_test, list(set(df_full["group"].tolist()))
        )

        # remove groups with too few examples from the dataset
        groups_to_remove = ["middle_east", "physical_dis"]
        df_full.drop(
            df_full[df_full["group"].isin(groups_to_remove)].index, inplace=True
        )
        df_test.drop(
            df_test[df_test["group"].isin(groups_to_remove)].index, inplace=True
        )

        # create the train and dev splits based on the group and bias label
        large_split, _ = train_test_split(
            df_full,
            train_size=COBRA_SPLIT_SIZE * 2,  # 4_000
            stratify=df_full[["group", "hasBiasedImplication"]],
            random_state=RANDOM_SEED[0],
        )
        df_train, df_dev = train_test_split(
            large_split,
            train_size=COBRA_SPLIT_SIZE,  # 2_000
            stratify=large_split[["group", "hasBiasedImplication"]],
            random_state=RANDOM_SEED[0],
        )

        data_dict = {}
        for split, df in zip(["train", "dev", "test"], [df_train, df_dev, df_test]):
            # build statements based of a template used in the original cobra frames paper and
            # compute corresponding md5 hash
            df = self._build_cobra_statements(df, split)
            df["text_hash"] = df["post"].apply(lambda x: md5(x.encode()).hexdigest())

            # change group to none if statement is not biased/offensive
            df.loc[df["hasBiasedImplication"] == 0, "group"] = "none"

            # find similar statements based on the sentence embeddings for few-shot prompting
            df["similar_texts_by_label"] = get_similar_texts_as_examples(
                df["post"].tolist(),
                df["hasBiasedImplication"].tolist(),
                df["text_hash"].tolist(),
                emb_model_path=self._emb_model,
                n=NUM_SIMILAR_EXAMPLES,
            )

            df.reset_index(drop=True, inplace=True)
            data_dict[split] = df

        return data_dict

    def _create_cobra_group_column(
        self, df: pd.DataFrame, groups: List[str]
    ) -> pd.DataFrame:
        """
        Create a group column for the test split by computing the most similar group
        based on the sentence embeddings of the statement + original target group column
        and the group names.

        Args:
            df (pd.DataFrame): The input dataframe containing Cobra Frames data.
            groups (List[str]): The list of group names.

        Returns:
            pd.DataFrame: The dataframe with the updated group column.
        """
        group_sentences = [f"The target group is: {group}" for group in groups]

        target_groups = df["targetGroup"].tolist()
        statements = df["statement"].to_list()

        target_group_with_context = []
        for idx, target_group in enumerate(target_groups):
            target_group_with_context.append(
                f"statement: {statements[idx]} target group: {target_group}"
            )

        # compute the sentence embeddings
        target_group_with_context_emb = compute_sentence_embeddings(
            target_group_with_context, self._emb_model
        )
        group_sentence_emb = compute_sentence_embeddings(
            group_sentences, self._emb_model
        )

        # get the best matches
        best_matches = self._get_best_cobra_group_matches(
            target_group_with_context_emb, group_sentence_emb, groups
        )

        df["group"] = best_matches
        return df

    def _get_best_cobra_group_matches(
        self, target_emb: np.ndarray, group_emb: np.ndarray, groups: List[str]
    ) -> List[str]:
        """
        Get the best matches for the Cobra group based on cosine similarities
        between target embeddings and group embeddings.

        Args:
            target_emb (np.ndarray): The target embeddings.
            group_emb (np.ndarray): The group embeddings.
            groups (List[str]): The list of group names.

        Returns:
            List[str]: The list of best matching group names.
        """
        if isinstance(target_emb, list):
            target_emb = np.array(target_emb)

        if isinstance(group_emb, list):
            group_emb = np.array(group_emb)

        cosine_similarities_groups = util.cos_sim(target_emb, group_emb)

        best_matches_groups = np.argmax(cosine_similarities_groups, axis=1)

        return [groups[idx] for idx in best_matches_groups]

    def _get_cobra_bias_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary bias/offensiveness labels for the Cobra Frames dataset by checking
        for specific words in the offensiveness column.

        Args:
            df (pd.DataFrame): The input dataframe containing Cobra Frames data.

        Returns:
            pd.DataFrame: The dataframe with the updated 'hasBiasedImplication' column,
            removing examples that could not be classified.
        """
        offensiveness = df["offensiveness"].tolist()
        binary_labels = []

        for label in offensiveness:
            label_word_list = (
                str(label)
                .lower()
                .replace(".", "")
                .replace("[", "")
                .replace(",", " ")
                .replace("/", " ")
                .split()
            )

            if set(label_word_list) & set(COBRA_NON_OFFENSIVE_HINTS):
                binary_labels.append(0)
                continue

            if set(label_word_list) & set(COBRA_OFFENSIVE_HINTS):
                binary_labels.append(1)
                continue

            binary_labels.append(2)

        df["hasBiasedImplication"] = binary_labels
        return df[
            df["hasBiasedImplication"] != 2
        ]  # remove examples that could not be classified

    def _build_cobra_statements(
        self, df: pd.DataFrame, split: str, template: str = COBRA_STATEMENT_TEMPLATE
    ) -> pd.DataFrame:
        """
        Build statements based on a template used in the original Cobra Frames paper
        and compute corresponding MD5 hash.

        Args:
            df (pd.DataFrame): The input dataframe containing Cobra Frames data.
            split (str): The data split ("train", "dev", or "test").
            template (str, optional): The template for creating statements. Defaults to
                COBRA_STATEMENT_TEMPLATE.

        Returns:
            pd.DataFrame: The dataframe with the built statements.
        """
        # the context column is named differently in the original test split from the cobra frames
        # paper, so we rename it to match the other splits
        if split == "test":
            df.rename(columns={"situationalContext": "speechContext"}, inplace=True)

        statements = []
        for _, row in df.iterrows():
            statement = template.format(
                speakerIdentity=row["speakerIdentity"],
                listenerIdentity=row["listenerIdentity"],
                speechContext=row["speechContext"],
                statement=row["statement"],
            )
            statements.append(statement)

        df["post"] = statements

        return df

    def __str__(self) -> str:
        dataset_status = {
            "sbic": "Loaded" if self.sbic_data is not None else "Not loaded",
            "stereoset": "Loaded" if self.stereoset_data is not None else "Not loaded",
            "cobra_frames": "Loaded" if self.cobra_frames is not None else "Not loaded",
        }
        return f"DataHandler: {dataset_status}"

    def __repr__(self) -> str:
        return self.__str__()
