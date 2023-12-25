import pandas as pd
import Levenshtein
import time


def divide_blocks_structured_keys(df1: pd.DataFrame, df2: pd.DataFrame, keys: list) -> dict:
    """
    This function iterates over the keys list and for each key, it divides both DataFrames into
    smaller blocks. Each block contains rows from the original DataFrame where the
    value in the specified key column is the same. This is done for each unique value
    found in the key column of the first DataFrame.

    :param df1: The first DataFrame to be divided into blocks.
    :param df2: The second DataFrame to be divided into blocks.
    :param keys: A list of column names in the DataFrames to be used for dividing into blocks
    :return:
    """
    block_keys = {}
    for column_name in keys:
        # Get unique values in the specified column
        unique_values = df1[column_name].unique()

        # Create two dictionaries for each dataset to store blocks
        block1 = {value: df1[df1[column_name] == value].reset_index(drop=True) for value in unique_values}
        block2 = {value: df2[df2[column_name] == value].reset_index(drop=True) for value in unique_values}

        # Store these dictionaries in the main block_keys dictionary
        block_keys[column_name] = (block1, block2)
    return block_keys


def row_matching(blocks: dict, column_name: str, similarity_threshold: float, similarity_metric: str) -> list:
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

    start = time.time()

    # check if the given column name is valid
    if column_name not in blocks:
        raise ValueError("Invalid column name")
    similar_rows = []

    # get the two blocks for given column name
    block1, block2 = blocks[column_name]

    for value in block1:
        if value in block2:
            key1 = block1[value]  # DataFrame in block1 associated with the current value
            key2 = block2[value]  # DataFrame in block2 associated with the same value
            for _, row1 in key1.iterrows():  # Iterate through each row in the DataFrame from block1
                for _, row2 in key2.iterrows():  # Iterate through each row in the DataFrame from block2
                    # Calculate the similarity between the current rows from block1 and block2
                    similarity = calculate_similarity(row1, row2, "Title", similarity_metric)
                    if similarity >= similarity_threshold:  # Check if similarity meets or exceeds the threshold
                        similar_rows.append(row1)  # Append the row from block1 to similar_rows if the condition is met

    # for duration of the process
    end = time.time()
    duration = end - start
    print(f"Duration {similarity_metric} : %f" % duration, "seconds", f"with {len(similar_rows)} matches")
    return similar_rows


def baseline_pipeline(df1: pd.DataFrame, df2: pd.DataFrame, similarity_threshold: float, similarity_metric: str) -> list:
    """
    This function iterates over every row in df1 and df2, computing the similarity between each pair of rows.
    Rows from df1 that are similar to any row in df2, based on the specified similarity threshold and metric,
    are collected and returned.

    Notes:
    The total duration of the process and the number of similar rows found are printed.

    :param df1: The first DataFrame
    :param df2: The second DataFrame
    :param similarity_threshold: The threshold for considering rows similar
    :param similarity_metric: The metric used to calculate similarity
    :return:
    """

    start = time.time()
    similar_rows = []

    for _, row1 in df1.iterrows():
        for _, row2 in df2.iterrows():
            similarity = calculate_similarity(row1, row2, "Title", similarity_metric)
            if similarity >= similarity_threshold:
                similar_rows.append(row1)

    # for duration of the process
    end = time.time()
    duration = end - start
    print(f"Duration Baseline : %f" % duration, "seconds", f"with {len(similar_rows)} matches")
    return similar_rows


def calculate_similarity(row1: pd.Series, row2: pd.Series, key: str, similarity_metric: str) -> float:
    """
    This function computes the similarity between two pandas Series (rows) based on the value in a specified key column.
    It supports different similarity metrics like 'Levenshtein' and 'Jaccard'. For 'Levenshtein', it calculates the
    normalized Levenshtein distance, and for 'Jaccard', it computes the Jaccard similarity score.

    :param row1:  The first row for similarity comparison.
    :param row2:  The second row for similarity comparison.
    :param key: The column key to use for extracting values from the rows (example: Years)
    :param similarity_metric: The metric to use for calculating similarity ('Levenshtein' or 'Jaccard')

    Notes: Further distance metrics can be implemented in this function

    :return: float: The similarity score between the two rows based on the specified metric.
    """
    if len(row1) != len(row2):
        raise ValueError("Both rows must have the same number of columns")
    similarity_score = 0
    if similarity_metric == 'Levenshtein':
        text1 = str(row1[key])
        text2 = str(row2[key])
        distance = Levenshtein.distance(text1, text2)
        similarity_score = 1 - (distance / max(len(str(row1[key])), len(str(row2[key]))))
    elif similarity_metric == "Jaccard":
        set1 = set(str(row1[key]))
        set2 = set(str(row2[key]))
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        similarity_score = intersection / union
    return similarity_score


if __name__ == '__main__':
    df_acm = pd.read_csv('Data_filtered/ACM 1995 2004.csv')
    df_dblp = pd.read_csv('Data_filtered/DBLP 1995 2004.csv')
    key_blocks = divide_blocks_structured_keys(df_acm, df_dblp, ['Year'])
    similar_jacquard = row_matching(key_blocks, "Year",
                                    similarity_threshold=0.9,
                                    similarity_metric='Jaccard')
    similar_levenshtein = row_matching(key_blocks, "Year",
                                       similarity_threshold=0.9,
                                       similarity_metric='Levenshtein')
    similar_baseline = baseline_pipeline(df_acm, df_dblp,
                                         similarity_threshold=0.9,
                                         similarity_metric='Jaccard')
