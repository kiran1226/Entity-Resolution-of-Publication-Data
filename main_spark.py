from pyspark.sql import SparkSession

from data_acquisition_spark import *

if __name__ == '__main__':
    spark = SparkSession.builder.appName("EntityResolution").getOrCreate()
    df_acm_spark = read_data_spark("Data/citation-acm-v8.txt", spark)
    df_dblp_spark = read_data_spark("Data/dblp.txt", spark)
    spark.stop()