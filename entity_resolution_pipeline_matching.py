import pandas as pd
import Levenshtein
import time

from pandas import Series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

from entity_resolution_pipeline_blocking import divide_blocks_structured_keys

# global variable
global_vectorizer = TfidfVectorizer()


def row_matching_structured_keys(blocks: dict, column_name: str, similarity_threshold: float,
                                 similarity_metric: str) -> list:
    """
    Compares rows within blocks of DataFrames based on a specified column,
    identifying rows that are similar according to a defined threshold and metric

    :param blocks: A dictionary where keys are column names and values are tuples of DataFrames.
                   Each tuple contains two DataFrames with rows grouped by the key's value.
    :param column_name: The name of the column to be used for identifying similar rows within the blocks.
    :param similarity_threshold: The threshold value for the similarity score.
                                Rows with a similarity score above this threshold are considered similar.
    :param similarity_metric: The name of the metric used to calculate the similarity between rows.

    :return: A list of rows from the first block of DataFrames that have similar counterparts in the second block,
          determined by the specified similarity metric and threshold.
    """

    print(f"Structured Keys row matching {similarity_metric} started!")
    start = time.time()

    # check if the given column name is valid
    if column_name not in blocks:
        print(blocks.values())
        raise ValueError("Invalid column name")
    similar_pairs = []

    # get the two blocks for given column name
    block1, block2 = blocks[column_name]

    for value in block1:
        if value in block2:
            key1 = block1[value]  # DataFrame in block1 associated with the current value
            key2 = block2[value]  # DataFrame in block2 associated with the same value
            for _, row1 in key1.iterrows():  # Iterate through each row in the DataFrame from block1
                for _, row2 in key2.iterrows():  # Iterate through each row in the DataFrame from block2
                    # Calculate the similarity between the current rows from block1 and block2
                    similarity = calculate_similarity(row1=row1,
                                                      row2=row2,
                                                      similarity_metric=similarity_metric,
                                                      key_columns=["Title", "Authors"])
                    if similarity >= similarity_threshold:  # Check if similarity meets or exceeds the threshold
                        similar_pairs.append((row1, row2))

    # for duration of the process
    end = time.time()
    duration = end - start
    print(f"Structured Keys row matching finished! "
          f"Duration {similarity_metric} : %f" % duration, "seconds", f"with {len(similar_pairs)} matches")
    return similar_pairs


def row_matching_ngrams(blocks: dict, similarity_threshold: float, similarity_metric: str) -> list:
    """
    Compares rows within n-gram blocks of DataFrames, identifying rows that are similar
    according to a defined threshold and metric.

    :param blocks: A dictionary with keys as n-grams and values as tuples of DataFrames.
    :param similarity_threshold: The threshold value for the similarity score.
    :param similarity_metric: The name of the metric used to calculate the similarity between rows.

    :return: A list of tuples, each containing a pair of similar rows.
    """

    print(f"N grams row matching {similarity_metric} started!")
    start = time.time()

    similar_pairs = []
    processed_pairs = set()  # Set to keep track of processed pairs

    for n_gram, (block1, block2) in blocks.items():
        for idx1, row1 in block1.iterrows():
            for idx2, row2 in block2.iterrows():
                # Create a tuple of indices or unique identifiers
                pair = (idx1, idx2)

                # Check if the pair has already been processed
                if pair in processed_pairs:
                    continue
                similarity = calculate_similarity(row1=row1, row2=row2, key_columns=["Title", "Authors"],
                                                  similarity_metric=similarity_metric)
                if similarity >= similarity_threshold:
                    similar_pairs.append((row1, row2))
                    processed_pairs.add(pair)

    # for duration of the process
    end = time.time()
    duration = end - start
    print(f"N grams row matching {similarity_metric} finished! "
          f"Duration {similarity_metric} : %f" % duration, "seconds", f"with {len(similar_pairs)} matches")

    return similar_pairs


def calculate_similarity(row1: pd.Series, row2: pd.Series, similarity_metric: str,
                         key_columns: list[str]) -> float:
    """
    This function computes the similarity between two pandas Series (rows) based on the value in a specified key column.
    It supports different similarity metrics like 'Levenshtein' and 'Jaccard'. For 'Levenshtein', it calculates the
    normalized Levenshtein distance, and for 'Jaccard', it computes the Jaccard similarity score.

    :param row1:  The first row for similarity comparison.
    :param row2:  The second row for similarity comparison.
    :param key_columns: important columns for row comparison
    :param similarity_metric: The metric to use for calculating similarity ('Levenshtein' or 'Jaccard')

    Notes: Further distance metrics can be implemented in this function

    :return: float: The similarity score between the two rows based on the specified metric.
    """

    # if both rows does not have the same "Year column" or dont have the same key columns
    # consider as they are not a match
    if any([
        row1["Year"] != row2["Year"],
        not all(key in row1 for key in key_columns),  # Check each key in row1
        not all(key in row2 for key in key_columns)  # Check each key in row2
    ]):
        return 0

    # Focus only on key columns for comparison
    row1_key = row1[key_columns].astype(str)
    row2_key = row2[key_columns].astype(str)

    if similarity_metric == 'Levenshtein':
        text1 = ' '.join(row1_key)
        text2 = ' '.join(row2_key)
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1  # Prevent division by zero
        return 1 - (Levenshtein.distance(text1, text2) / max_len)  # Normalized
    elif similarity_metric == "Jaccard":
        set1 = set(' '.join(row1_key))
        set2 = set(' '.join(row2_key))
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0
    elif similarity_metric == "TF-IDF":
        tfidf_matrix = global_vectorizer.transform([' '.join(row1_key), ' '.join(row2_key)])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return score[0][0]
    return 0


def calculate_metrics(base_pairs: list[tuple], comparison_pairs: list[tuple]):
    """
    This function compares the base pipeline and ER pipelines and returns precision, recall, f1_score
    :param base_pairs:
    :param comparison_pairs:
    :return:
    """

    def to_hashable(pair):
        return tuple(pair[0].values), tuple(pair[1].values)

    # Convert lists of pairs of pd.Series to sets of hashable tuples
    base_pairs_set = set(map(to_hashable, base_pairs))
    comparison_pairs_set = set(map(to_hashable, comparison_pairs))

    # Calculate True Positives, False Positives, and False Negatives
    true_positives = len(base_pairs_set & comparison_pairs_set)
    false_positives = len(comparison_pairs_set - base_pairs_set)
    false_negatives = len(base_pairs_set - comparison_pairs_set)

    # Calculate Precision, Recall, and F1 Score
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score


