import pandas as pd
import Levenshtein
import time

from pandas import Series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
            return 0  # Prevent division by zero
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


def write_series_to_csv(series_list: list[pd.Series], file_name: str = "Matched Entities.csv"):
    """
    The function writes a list of pd.Series into a csv

    :param series_list:
    :param file_name:
    :return:
    """

    # Combine all Series into a DataFrame
    df = pd.DataFrame(series_list)

    # Write DataFrame to CSV
    df.to_csv(file_name, index=False)


if __name__ == '__main__':
    print(time.time())
    df_acm = pd.read_csv('Data_filtered/ACM 1995 2004.csv').head(500)
    df_dblp = pd.read_csv('Data_filtered/DBLP 1995 2004.csv').head(500)

    # Prepare text data for fitting the vectorizer
    text_data_acm = df_acm['Title'].fillna('') + " " + df_acm['Authors'].fillna('')
    text_data_dblp = df_dblp['Title'].fillna('') + " " + df_dblp['Authors'].fillna('')
    combined_text_data = pd.concat([text_data_acm, text_data_dblp]).unique()

    # Fit the global vectorizer
    global_vectorizer.fit(combined_text_data)

    """
    
    similar_baseline = baseline_pipeline(df_acm, df_dblp,
                                         similarity_threshold=0.9,
                                         similarity_metric='Jaccard')
    """

    # N grams for the column "Authors"
    key_blocks_n_grams = divide_blocks_n_gram_blocking(df1=df_acm, df2=df_dblp, column="Authors", n=10)
    similar_tf_idf_n_grams = row_matching_ngrams(blocks=key_blocks_n_grams,
                                                 similarity_threshold=0.9,
                                                 similarity_metric='TF-IDF')
    similar_jacquard_n_grams = row_matching_ngrams(blocks=key_blocks_n_grams,
                                                   similarity_threshold=0.9,
                                                   similarity_metric='Jaccard')
    similar_levenshtein_n_grams = row_matching_ngrams(key_blocks_n_grams,
                                                      similarity_threshold=0.9,
                                                      similarity_metric='Levenshtein')

    # Structured Keys for the column "Year"
    key_blocks_structured_keys = divide_blocks_structured_keys(df1=df_acm,
                                                               df2=df_dblp,
                                                               column_name="Year")
    similar_jacquard_structured_keys = row_matching_structured_keys(blocks=key_blocks_structured_keys,
                                                                    column_name="Year",
                                                                    similarity_threshold=0.9,
                                                                    similarity_metric='Jaccard')
    similar_levenshtein_structured_keys = row_matching_structured_keys(blocks=key_blocks_structured_keys,
                                                                       column_name="Year",
                                                                       similarity_threshold=0.9,
                                                                       similarity_metric='Levenshtein')
