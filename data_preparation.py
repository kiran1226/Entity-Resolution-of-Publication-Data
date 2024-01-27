import pandas as pd


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    filters the given dataframe by all the publications published between 1995 and 2004 and have
    'VLDB' and 'SIGMOD' as venues

    :param df: the dataframe to filter
    :return: filtered dataframe
    """

    # Collect all the publications published between 1995 and 2004
    df = df[(df['Year'] >= 1995) & (df['Year'] <= 2004)]

    # Collect all the publications published between 1995 and 2004 in 'VLDB' and 'SIGMOD' venues
    df = df[df['Publication Venue'].str.contains('VLDB|SIGMOD', case=False, na=False)]
    return df.reset_index()
