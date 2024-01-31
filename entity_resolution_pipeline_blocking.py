import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer


# global variable
global_vectorizer = TfidfVectorizer()


def divide_blocks_structured_keys(df1: pd.DataFrame, df2: pd.DataFrame, column_name: str) -> dict:
    """
    This function iterates over the keys list and for each key, it divides both DataFrames into
    smaller blocks. Each block contains rows from the original DataFrame where the
    value in the specified key column is the same. This is done for each unique value
    found in the key column of the first DataFrame.

    :param df1: The first DataFrame to be divided into blocks.
    :param df2: The second DataFrame to be divided into blocks.
    :param column_name:column used for dividing into blocks
    :return:
    """

    print("Structured Keys Blocking started!")
    start = time.time()

    block1 = {key: group.reset_index(drop=True) for key, group in df1.groupby(column_name)}
    block2 = {key: group.reset_index(drop=True) for key, group in df2.groupby(column_name)}

    block_keys = {column_name: (block1, block2)}

    end = time.time()
    print(f"Structured Keys Blocking finished! "
          f"Duration Structured Keys Blocking: {end - start} seconds with {len(block1.keys())} blocks")
    return block_keys


def divide_blocks_n_gram_blocking(df1: pd.DataFrame, df2: pd.DataFrame, column: str, n: int) -> dict:
    """
    This function creates blocks based on n-grams of a specified column.

    :param df1: The first DataFrame.
    :param df2: The second DataFrame.
    :param column: The column name to create n-grams from.
    :param n: The length of the n-gram.
    :return: A dictionary with keys as n-grams and values as tuples of DataFrames.
    """

    print("N Grams Blocking started!")
    start = time.time()
    df1['n_grams'] = df1[column].apply(lambda x: generate_n_grams(str(x), n))
    df2['n_grams'] = df2[column].apply(lambda x: generate_n_grams(str(x), n))

    unique_n_grams = set().union(*df1['n_grams'], *df2['n_grams'])

    block_keys = {}
    for n_gram in unique_n_grams:
        block1 = df1[df1['n_grams'].apply(lambda x: n_gram in x)]
        block2 = df2[df2['n_grams'].apply(lambda x: n_gram in x)]
        block_keys[n_gram] = (block1, block2)

    # for duration of the process
    end = time.time()
    duration = end - start
    print(f"N Grams Blocking finished! Duration: %f"
          % duration, "seconds", f"with {len(block_keys)} blocks")
    return block_keys


def generate_n_grams(text, n):
    """
    helper function for n_gram blocking
    :param text:
    :param n:
    :return:
    """
    return [text[i:i + n] for i in range(len(text) - n + 1)]
