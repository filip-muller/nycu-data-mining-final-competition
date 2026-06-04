import csv
import sys

INPUT_FILE = sys.argv[1]
MULTIPLE = float(sys.argv[2])
THRESHOLD_FOR_ZERO = float(sys.argv[3])
CLOSENESS_THRESHOLD = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05

OUTPUT_FILE = f"{INPUT_FILE.split(".csv")[0]}_{MULTIPLE}_{THRESHOLD_FOR_ZERO}_{CLOSENESS_THRESHOLD}.csv"

total_discarded_value = 0
capped_cells = 0
affected_regions = 0

closeness_discarded_value = 0
clonesess_modified_cells = 0

with open(INPUT_FILE, newline="") as infile, open(OUTPUT_FILE, "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        new_row = []
        region_affected = False
        for cell in row:
            try:
                res = float(cell) * MULTIPLE
                if res < THRESHOLD_FOR_ZERO:
                    total_discarded_value += res
                    capped_cells += 1
                    res = 0
                    if not region_affected:
                        affected_regions += 1
                        region_affected = True
                # make preds close to an integer that integer (dont double count the ones zerod before)
                elif abs(res - round(res)) < CLOSENESS_THRESHOLD:
                    closeness_discarded_value += abs(res - round(res))
                    clonesess_modified_cells += 1
                    res = round(res)
                new_row.append(res)
            except ValueError:
                new_row.append(cell)  # keep non-numeric cells (like headers) as-is
        writer.writerow(new_row)

print(f"Thresholding removed {total_discarded_value:.2f} in total, or {total_discarded_value / (2248 * 5):.5f} per value")
print(f"Affected regions: {affected_regions}, affected cells: {capped_cells}")
print(f"Closeness threshold discarded {closeness_discarded_value:.2f} value and modified {clonesess_modified_cells} cells")
print(f"Done! Multiplied all values by {MULTIPLE} and zeroed values under {THRESHOLD_FOR_ZERO} → {OUTPUT_FILE}")
