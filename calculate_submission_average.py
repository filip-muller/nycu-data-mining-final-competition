import csv
import sys


INPUT_FILE = sys.argv[1]

columns = {}

with open(INPUT_FILE, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        for key, val in row.items():
            if key == "region_id":
                continue
            columns.setdefault(key, []).append(float(val))

all_values = []
print("Per-column averages:")
for col, values in columns.items():
    avg = sum(values) / len(values)
    all_values.extend(values)
    print(f"  {col}: {avg:.4f}")

overall = sum(all_values) / len(all_values)
print(f"\nOverall average: {overall:.4f}")
