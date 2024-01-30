import pandas as pd
import time
import codecs
import numpy as np


def read_data(path: str, encoding: str) -> pd.DataFrame:
    """
    Reads the given file according to the encoding line by line and returns a dataframe and measures the duration of the
    function

    :param path: the path of the file
    :param encoding: the encoding of the file
    :return:  dataframe of the read file
    """

    print("Reading data started!")
    start = time.time()
    current_paper = {'Paper ID': '', 'Title': '', 'Authors': '', 'Publication Venue': '', 'Year': np.nan}
    data = []

    # read file line by line add values to the columns in the df
    with codecs.open(path, "r", encoding) as file:
        for line in file:
            line = line.strip()
            if not line:
                data.append(current_paper)
                current_paper = {'Paper ID': '', 'Title': '', 'Authors': '', 'Publication Venue': '', 'Year': np.nan}
            elif line.startswith("#*"):
                current_paper['Title'] = line[2:]
            elif line.startswith("#@"):
                current_paper['Authors'] = line[2:]
            elif line.startswith("#t"):
                year = line[2:len(line)]
                if year is not np.nan:
                    current_paper['Year'] = int(year)
            elif line.startswith("#c"):
                current_paper['Publication Venue'] = line[2:]
            elif line.startswith("#index"):
                current_paper['Paper ID'] = line[6:]

    # turn list data into dataframe
    df = pd.DataFrame(data)

    # drop null values
    df.dropna(subset=['Year', 'Publication Venue'], inplace=True)

    # convert year from string to int
    df['Year'] = df['Year'].astype(int)

    # measure the function duration for the given filename
    filename = path.split("/")[1]
    end = time.time()
    duration = end - start
    print(f"Reading data finished -- Spark! Duration {filename}: %f" % duration, "seconds")
    return df
