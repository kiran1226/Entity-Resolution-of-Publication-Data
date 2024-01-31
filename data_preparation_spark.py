from pyspark.sql import DataFrame


def filter_data_spark(df: DataFrame) -> DataFrame:
    """
    Filters the given Spark DataFrame by all the publications published between 1995 and 2004 and have
    'VLDB' and 'SIGMOD' as venues.

    :param df: the Spark DataFrame to filter
    :return: filtered Spark DataFrame
    """

    # Filter publications published between 1995 and 2004
    df_filtered = df.filter((df['Year'] >= 1995) & (df['Year'] <= 2004))

    # Filter publications in 'VLDB' and 'SIGMOD' venues
    df_filtered = df_filtered.filter(df_filtered['Publication Venue'].rlike('VLDB|SIGMOD'))

    return df_filtered
