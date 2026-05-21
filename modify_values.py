import csv
import sys

INPUT_FILE = sys.argv[1] #"submission_monthly_average.csv"
MULTIPLE = float(sys.argv[2]) #1.4  # Change this to whatever you want
OUTPUT_FILE = f"{INPUT_FILE.split(".csv")[0]}_{MULTIPLE}.csv"

with open(INPUT_FILE, newline="") as infile, open(OUTPUT_FILE, "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        new_row = []
        for cell in row:
            try:
                new_row.append(float(cell) * MULTIPLE)
            except ValueError:
                new_row.append(cell)  # keep non-numeric cells (like headers) as-is
        writer.writerow(new_row)

print(f"Done! Multiplied all values by {MULTIPLE} → {OUTPUT_FILE}")
