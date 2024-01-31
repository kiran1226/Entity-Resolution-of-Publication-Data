from data_preparation_spark import *
from data_acquisition_spark import *
from entity_resolution_pipeline_blocking_spark import *
from entity_resolution_pipeline_matching_spark import *
from csv_functions import *

if __name__ == '__main__':
    # create a SparkSession called EntityResolution
    spark = SparkSession.builder.appName("EntityResolution").getOrCreate()

    # read data with the spark function
    df_acm_spark = read_data_spark("Data/citation-acm-v8.txt", spark)
    df_dblp_spark = read_data_spark("Data/dblp.txt", spark)

    # filter data with spark functions
    df_acm_spark_filtered = filter_data_spark(df_acm_spark)
    df_dblp_spark_filtered = filter_data_spark(df_dblp_spark)

    block_keys_dict = divide_blocks_structured_keys_spark(df1=df_acm_spark_filtered,
                                                          df2=df_dblp_spark_filtered,
                                                          column_name="Year")

    similar_pairs = row_matching_structured_keys_spark(blocks=block_keys_dict,
                                                       column_name="Year",
                                                       similarity_metric="Levenshtein",
                                                       similarity_threshold=0.9)

    print(similar_pairs)
    # stop spark session
    spark.stop()
