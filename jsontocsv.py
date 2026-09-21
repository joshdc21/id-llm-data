import json
import csv

def jsonl_to_csv(jsonl_filepath, csv_filepath, encoding="utf-8"):
    # Open JSONL file with the specified encoding
    with open(jsonl_filepath, "r", encoding=encoding) as json_file:
        
        # Read each line as a separate JSON object
        data = [json.loads(line) for line in json_file if line.strip()]

    # Open CSV file for writing
    with open(csv_filepath, "w", newline="", encoding="utf-8-sig") as csv_file:
        
        # Get headers from the first JSON object
        headers = data[0].keys()

        # Create CSV writer
        csv_writer = csv.DictWriter(csv_file, fieldnames=headers)

        # Write headers
        csv_writer.writeheader()

        # Write data
        csv_writer.writerows(data)


# Example usage
jsonl_filepath = "articles.jsonl"
csv_filepath = "articles.csv"

jsonl_to_csv(jsonl_filepath, csv_filepath)

print(
    f"JSONL data from '{jsonl_filepath}' has been successfully "
    f"converted to CSV format in '{csv_filepath}'."
)