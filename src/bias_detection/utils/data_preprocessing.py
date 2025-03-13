import html
import random
import re
import warnings
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


def remove_multiple_whitespace(s: str) -> str:
    """
    Remove multiple consecutive whitespace characters (including spaces,
    tabs, and newlines) from the input string and replace them with a single space.

    Args:
        s (str): The input string containing multiple whitespace characters.

    Returns:
        str: A new string with multiple consecutive whitespace characters replaced
        by a single space.
    """
    return re.sub(r"\s+", " ", s)


def replace_links(s: str, place_holder: str = "[LINK]") -> str:
    """
    Replace links (http or https) and  from a given string with a placeholder.

    Args:
        s (str): The input string containing text that may include links.
        place_holder (str): The placeholder to replace the links with.
            Defaults to "[LINK]".

    Returns:
        str: A modified string with links replaced by the placeholder.
    """
    link_pattern = re.compile(
        r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|(pic\.twitter\.com/\S+)"
    )
    return link_pattern.sub(place_holder, s).strip()


def remove_mentions(s: str) -> str:
    """
    Remove mentions (usernames starting with '@') from a given string.

    Args:
        s (str): The input string containing mentions.

    Returns:
        str: A modified string with mentions removed.
    """
    pattern = re.compile(r"@\w+")
    return pattern.sub(r"", s).strip()


def clean_string(
    s: str, replace_link: bool = True, remove_mention: bool = False
) -> str:
    """
    Clean and preprocess a given string.

    Args:
        s (str): The input string.
        replace_link (bool): Whether to replace links in the input string.
        remove_mention (bool): Whether to remove mentions in the input string.

    Returns:
        str: A cleaned and processed version of the input string.
    """
    if not isinstance(s, str):
        print(f"Input string '{s}' is not a string but {type(s)}")

    s = s.replace("\n", " ")
    s = s.replace("&amp;", "&")
    s = html.unescape(s)

    if replace_link:
        s = replace_links(s, place_holder="[LINK]")

    if remove_mention:
        s = remove_mentions(s)

    # remove certain special characters found in SBIC
    chars_to_remove = ["`", '"', "“", "”", "\u200f", "*", "_", "-"]
    for c in chars_to_remove:
        s = s.replace(c, "")

    return remove_multiple_whitespace(s).strip()


def compute_sentence_embeddings(
    input_strings: List[str], model_path: str
) -> List[np.ndarray]:
    """
    Compute embeddings for a list of input sentences using a pre-trained
    SentenceTransformer model.

    Args:
        input_strings (List[str]): List with input sentences to compute embeddings for.
        model_path (str): Path to a pre-trained SentenceTransformer model.

    Returns:
        List[np.ndarray]: List of  NumPy arrays where each row is an embedding
            corresponding to a sentence in the input list.
    """
    if not isinstance(input_strings, list):
        input_strings = [input_strings]

    if model_path is None:
        raise ValueError(
            "model_path is None. Provide a valid path to a SentenceTransformer model."
        )

    model = SentenceTransformer(model_path)
    return list(model.encode(input_strings))


def get_similar_texts_as_examples(
    text_data: List[str],
    labels: List[str],
    md5_hashes: List[str],
    emb_model_path: str,
    n: int,
) -> List[Dict[Any, List[str]]]:
    """
    Generate a list of dictionaries mapping each label to n MD5 hashes of text data
    that are most similar to each text, excluding the text itself.

    Args:
        text_data (List[str]): List of text strings to be analyzed.
        labels (List[str]): Corresponding labels for each text string.
        md5_hashes (List[str]): MD5 hashes for each text string.
        emb_model_path (str): Path to the pre-trained embedding model file.
        n (int): Number of similar texts to retrieve for each label.

    Returns:
        List[Dict[Any, List[str]]]: A list of dictionaries for each text, where each
                                    dictionary maps labels to lists of n MD5 hashes
                                    of the most similar texts under the same label.
    """
    if not isinstance(labels[0], str):
        labels = [str(label) for label in labels]

    embeddings = compute_sentence_embeddings(text_data, emb_model_path)

    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -1)  # exclude self-comparison

    # find top 5 most similar text samples for a given sample for each label
    # samples are stored as MD5 hashes
    most_similar_hashes_by_label = []
    for idx in range(len(text_data)):
        most_similar_hashes = {}
        for category in set(labels):
            category_indices = [i for i, lab in enumerate(labels) if lab == category]
            category_sims = sim_matrix[idx, category_indices]
            # get indices of top n values
            top_indices = np.argsort(category_sims)[-n:][::-1]
            most_similar_hashes[category] = [
                md5_hashes[category_indices[i]] for i in top_indices
            ]

        most_similar_hashes_by_label.append(most_similar_hashes)

    return most_similar_hashes_by_label


def create_balanced_subset(
    df: pd.DataFrame, subset_size: int, seed: int
) -> pd.DataFrame:
    """
    Create a balanced subset of a dataframe with equal representation of categories.

    Args:
        df (pd.DataFrame): The input dataframe containing SBIC data.
        subset_size (int): The desired size of the subset.
        seed (int): The random seed for reproducibility.

    Raises:
        ValueError: If there aren't enough samples for a given category.
        ValueError: If there are duplicate posts in the result.

    Returns:
        pd.DataFrame: The balanced subset dataframe.
    """
    random.seed(seed)

    # find all categories
    all_categories = [
        cat
        for cat in list(df["targetCategory"].explode().unique())
        if isinstance(cat, str)
    ]

    # calculate how many samples are needed for each category and label
    # also find the number of samples to fill the subset to the desired size
    half_sample_size = subset_size // 2
    category_sample_size = subset_size // (2 * len(all_categories))
    remaining_samples = half_sample_size - (category_sample_size * len(all_categories))

    df_label_1 = df[df["hasBiasedImplication"] == 1]
    df_label_0 = df[df["hasBiasedImplication"] == 0]

    # first select the remaining samples and remove them from the original dataframe
    df_remaining_samples = df_label_1.sample(n=remaining_samples, random_state=seed)
    df_label_1 = df_label_1.drop(df_remaining_samples.index)

    # make list of indices for every category
    idx_lists = defaultdict(list)
    for idx, sample in enumerate(df_label_1["targetCategory"].tolist()):
        for cat in all_categories:
            if cat in sample:
                idx_lists[cat].append(idx)

    sorted_idx_lists = dict(sorted(idx_lists.items(), key=lambda item: len(item[1])))

    # sample from all categories
    chosen_posts = set()
    for cat, indices in sorted_idx_lists.items():
        purged_idx = [
            x for x in indices if df_label_1["post"].iloc[x] not in chosen_posts
        ]

        if len(purged_idx) >= category_sample_size:
            sampled_idx = random.sample(purged_idx, category_sample_size)
            chosen_posts.update(df_label_1["post"].iloc[x] for x in sampled_idx)
        else:
            raise ValueError(
                f"For category {cat} are not enough samples left ({purged_idx})."
                f" Required are {category_sample_size}"
            )

    df_part_1 = df_label_1[df_label_1["post"].isin(chosen_posts)]

    # sample from label 0 excluding already chosen posts
    df_part_2 = df_label_0[~df_label_0["post"].isin(chosen_posts)].sample(
        n=half_sample_size, random_state=seed
    )

    df_combined = pd.concat(
        [df_part_1, df_part_2, df_remaining_samples], axis=0, ignore_index=True
    )

    if len(df_combined[df_combined["post"].duplicated()]) > 0:
        raise ValueError("There are duplicate posts in the result.")

    return df_combined


def remove_duplicate_posts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove posts that have conflicting labels.

    Args:
        df (pd.DataFrame): The input DataFrame with columns 'post' and 'hasBiasedImplication'.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df.drop_duplicates(subset=["post"], keep="first", inplace=True)
    label_counts = df.groupby("post")["hasBiasedImplication"].nunique()
    conflicting_posts = label_counts[label_counts > 1].index
    return df[~df["post"].isin(conflicting_posts)].copy()


def concat_sentences_stereoset(
    sentences: List[str], gold_labels: List[int], context: str
) -> List[str]:
    """
    Concatenates each sentence with a given context based on provided gold labels.

    Args:
        sentences (List[str]): A list of sentences to be concatenated with the context.
        gold_labels (List[int]): A list of integer labels indicating the position in the
                                 final list where each corresponding sentence should be placed.
        context (str): A string that will be concatenated to each sentence.

    Returns:
        List[str]: A list of concatenated sentences in the order specified by gold labels.
                   The list will always contain exactly three elements.

    Raises:
        ValueError: If any element in the final list is None, indicating a missing sentence.
    """
    final_sentences = [None, None, None]

    for idx, sentence in enumerate(sentences):
        final_sentences[gold_labels[idx]] = context + " " + sentence

    if None in final_sentences:
        raise ValueError(
            f"One or more positions in the final list are not filled: {final_sentences} "
            f"Check the gold labels and sentences. {gold_labels} {sentences}"
        )

    return final_sentences
