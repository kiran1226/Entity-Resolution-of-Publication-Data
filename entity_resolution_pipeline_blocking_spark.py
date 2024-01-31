from pyspark.sql import DataFrame
from pyspark.sql.functions import collect_list, struct
import time


def divide_blocks_structured_keys_spark(df1: DataFrame, df2: DataFrame, column_name: str) -> dict:
    """
    This function divides both DataFrames into smaller blocks based on a specified column.
    """

    print("Structured Keys Blocking --Spark started!")
    start = time.time()

    # Create a structure with all necessary fields
    paper_structure = struct("Paper ID", "Title", "Authors", "Publication Venue", "Year")

    # Group DataFrames by the specified 'column_name' and aggregate the paper details into a list for each group
    block1 = df1.groupBy(column_name).agg(
        collect_list(paper_structure).alias("paper_details")
    )
    block2 = df2.groupBy(column_name).agg(
        collect_list(paper_structure).alias("paper_details")
    )

    block_keys = {column_name: (block1, block2)}

    end = time.time()
    print(f"Structured Keys Blocking --Spark finished! Duration: {end - start} seconds")

    return block_keys

