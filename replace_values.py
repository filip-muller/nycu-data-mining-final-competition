import csv
import sys

INPUT_FILE = "data/sample_submission.csv"
REPLACE_VALUE = 4.0  # Change this to whatever you want
OUTPUT_FILE = f"sample_submission_{REPLACE_VALUE}.csv"

with open(INPUT_FILE, newline="") as infile, open(OUTPUT_FILE, "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        new_row = []
        for cell in row:
            try:
                new_row.append(REPLACE_VALUE if float(cell) == 0 else cell)
            except ValueError:
                new_row.append(cell)  # keep non-numeric cells (like headers) as-is
        writer.writerow(new_row)

print(f"Done! Zeros replaced with {REPLACE_VALUE} → {OUTPUT_FILE}")
