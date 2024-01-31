from pyspark.sql.functions import udf, col, explode, lit, coalesce
from pyspark.sql.types import FloatType
import time
import Levenshtein


# Define the UDFs
def levenshtein_similarity_udf():
    def levenshtein_similarity(text1: str, text2: str):
        if not text1 or not text2:
            return 0.0
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        return 1 - (Levenshtein.distance(text1, text2) / max_len)

    return udf(levenshtein_similarity, FloatType())


def jaccard_similarity_udf():
    def jaccard_similarity(text1: str, text2: str):
        set1 = set(text1.split()) if text1 else set()
        set2 = set(text2.split()) if text2 else set()
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0.0

    return udf(jaccard_similarity, FloatType())


# Register UDFs
levenshtein_udf = levenshtein_similarity_udf()
jaccard_udf = jaccard_similarity_udf()


def calculate_similarity_spark(title1: str, authors1: str, title2: str, authors2: str, similarity_metric: str):
    combined_text1 = (title1 or '') + ' ' + (authors1 or '')
    combined_text2 = (title2 or '') + ' ' + (authors2 or '')

    if similarity_metric == 'Levenshtein':
        return levenshtein_udf(combined_text1, combined_text2)
    elif similarity_metric == 'Jaccard':
        return jaccard_udf(combined_text1, combined_text2)
    else:
        return 0


calculate_similarity_udf = udf(calculate_similarity_spark, FloatType())


def row_matching_structured_keys_spark(blocks: dict, column_name: str,
                                       similarity_threshold: float, similarity_metric: str):
    print(f"Structured Keys row matching --Spark {similarity_metric} started!")
    start = time.time()

    block1, block2 = blocks[column_name]
    exploded_block1 = block1.select(column_name, explode("paper_details").alias("details1"))
    exploded_block2 = block2.select(column_name, explode("paper_details").alias("details2"))

    joined_df = exploded_block1.join(exploded_block2, on=column_name)

    # Apply similarity calculation
    if similarity_metric == 'Levenshtein':
        similarity_udf = levenshtein_udf
    elif similarity_metric == 'Jaccard':
        similarity_udf = jaccard_udf
    else:
        raise ValueError("Unknown similarity metric")

    combined_df = joined_df.withColumn(
        "combined_text1",
        coalesce(col("details1.Title"), lit("")) + lit(" ") + coalesce(col("details1.Authors"), lit(""))
    ).withColumn(
        "combined_text2",
        coalesce(col("details2.Title"), lit("")) + lit(" ") + coalesce(col("details2.Authors"), lit(""))
    ).withColumn(
        "similarity",
        similarity_udf(col("combined_text1"), col("combined_text2"))
    )

    print(combined_df.show())
    # Filter rows based on similarity threshold
    similar_pairs_df = combined_df.filter(col("similarity") >= similarity_threshold)

    # Collecting the results
    similar_pairs = [(row['details1']['Paper ID'], row['details2']['Paper ID']) for row in similar_pairs_df.collect()]

    end = time.time()
    print(f"Structured Keys row matching --Spark finished! Duration: {end - start} seconds,"
          f" with {len(similar_pairs)} matches")
    return similar_pairs
