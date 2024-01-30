from pyspark.sql import SparkSession
import time


def process_lines(iterator):
    current_paper = None  # Initialize current_paper as None

    for line in iterator:
        if line.startswith("#index"):
            if current_paper:
                yield current_paper  # Yield the previous paper record
            current_paper = {'Paper ID': line[6:], 'Title': '', 'Authors': '', 'Publication Venue': '', 'Year': None}
        elif current_paper is not None:  # Ensure current_paper is not None before assignment
            if line.startswith("#*"):
                current_paper['Title'] = line[2:]
            elif line.startswith("#@"):
                current_paper['Authors'] = line[2:]
            elif line.startswith("#t"):
                year = line[2:].strip()
                if year.isdigit():
                    current_paper['Year'] = int(year)
            elif line.startswith("#c"):
                current_paper['Publication Venue'] = line[2:]

    if current_paper:
        yield current_paper  # Yield the last paper record


def read_data_spark(path: str, spark: SparkSession):
    print("Reading data started!")
    start = time.time()

    # Reading the file into an RDD
    rdd = spark.sparkContext.textFile(path)

    # Process each partition of lines to extract fields
    processed_rdd = rdd.mapPartitions(process_lines)

    # Convert to DataFrame
    df = spark.createDataFrame(processed_rdd)

    # Drop null values and convert year to integer
    df = df.filter(df['Year'].isNotNull() & df['Publication Venue'].isNotNull())
    df = df.withColumn("Year", df["Year"].cast("integer"))

    end = time.time()

    filename = path.split("/")[1]
    print(f"Reading data finished! Duration {filename}: {end - start} seconds")
    return df

