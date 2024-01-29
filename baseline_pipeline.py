import pandas as pd

from entity_resolution_pipeline_blocking import *
import time
from entity_resolution_pipeline_matching import *


def baseline_pipeline(df1: pd.DataFrame, df2: pd.DataFrame, similarity_threshold: float, similarity_metric: str) \
        -> list[tuple[pd.Series, pd.Series]]:
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
    print("Baseline pipeline started!")
    start = time.time()

    key_blocks_structured_keys = divide_blocks_structured_keys(df1=df1,
                                                               df2=df2,
                                                               column_name="Year")
    similar_pairs = row_matching_structured_keys(blocks=key_blocks_structured_keys,
                                                 column_name="Year",
                                                 similarity_threshold=similarity_threshold,
                                                 similarity_metric=similarity_metric)

    # write similar rows into a file named Matched Entities_<similarity_metric>.csv
    # write_series_to_csv(series_list=similar_rows, file_name=f"Matched Entities {similarity_metric}.csv")

    # for duration of the process
    end = time.time()
    duration = end - start
    print(f"Duration Baseline : %f" % duration, "seconds", f"with {len(similar_pairs)} matches")
    return similar_pairs
