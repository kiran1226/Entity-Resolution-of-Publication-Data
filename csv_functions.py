import csv


def write_similar_pairs_to_csv(similar_pairs: list[tuple], file_name: str = "Matched Entities.csv"):
    """
        Write the pairs to a CSV file. Each pair is converted to a string and written as a row in the CSV.

        :param similar_pairs: List of tuples representing pairs.
        :param file_name: The name of the CSV file to write.
        """
    with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Convert each pair to a string and write as a row
        for pair in similar_pairs:
            writer.writerow([str(pair)])
