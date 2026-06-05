import csv
import sys
import os


dryrun = False
if "--dry" in sys.argv:
    dryrun = True
    sys.argv.remove("--dry")

use_manual_thresholds = False
if "--manual" in sys.argv:
    use_manual_thresholds = True
    sys.argv.remove("--manual")


INPUT_FILE = sys.argv[1]
MULTIPLE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
THRESHOLD_FOR_ZERO = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
CLOSENESS_THRESHOLD = float(sys.argv[4]) if len(sys.argv) > 4 else 0.49

# doesnt work yet, applied before scaling
# MANUAL_THRESHOLDS = [0.95, 1.6, 2.3, 3.2, 4.0]
# MANUAL_THRESHOLDS = [0.6666, 1.55, 2.3, 3.2, 4.0]
MANUAL_THRESHOLDS = [0.6666, 1.65, 2.55, 3.4, 4.0]

OUTPUT_FILE = f"{INPUT_FILE.split(".csv")[0]}_{MULTIPLE}_{THRESHOLD_FOR_ZERO}_{CLOSENESS_THRESHOLD}.csv"
if use_manual_thresholds:
    OUTPUT_FILE = f"{INPUT_FILE.split(".csv")[0]}_M.csv"

print("Stats before:")
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
print(f"Overall average: {overall:.4f}")

print("Distribution:")
for i in range(10):
    threshold = 0.5 * i
    upper = threshold + 0.5
    vals_between = len([v for v in all_values if threshold < v <= upper])
    print(f"{threshold}-{upper}: {vals_between / len(all_values) * 100:.3f}%")


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
                if use_manual_thresholds:
                    v = float(cell)
                    thresholds = list(MANUAL_THRESHOLDS)
                    # placeholder at the end for a cleaner for cycle
                    thresholds.append(10000)
                    for i, th in enumerate(thresholds):
                        if v < th:
                            res = i
                            break
                else:
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

print("Stats after:")
columns = {}

vals = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    "unrounded": 0,
}

with open(OUTPUT_FILE, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        for key, val in row.items():
            if key == "region_id":
                continue
            flt = float(val)
            if flt == int(flt):
                # thresholded to be an int
                intgr = int(flt)
                vals[intgr] += 1
            else:
                # not rounded
                vals["unrounded"] += 1
            columns.setdefault(key, []).append(float(val))

all_values = []
print("Per-column averages:")
for col, values in columns.items():
    avg = sum(values) / len(values)
    all_values.extend(values)
    print(f"  {col}: {avg:.4f}")

overall = sum(all_values) / len(all_values)
print(f"\nOverall average: {overall:.4f}")

total_vals = sum(vals.values())
print("Distribution:")
for k, v in vals.items():
    print(f"{k}: {v / total_vals * 100:.2f}%")

if not use_manual_thresholds:
    print(f"Thresholding removed {total_discarded_value:.2f} in total, or {total_discarded_value / total_vals:.5f} per value")
    print(f"Affected regions: {affected_regions}, affected cells: {capped_cells}")
    print(f"Closeness threshold discarded {closeness_discarded_value:.2f} value and modified {clonesess_modified_cells} cells")
    print(f"Done! Multiplied all values by {MULTIPLE} and zeroed values under {THRESHOLD_FOR_ZERO} and rounded values closer than {CLOSENESS_THRESHOLD} → {OUTPUT_FILE}")
else:
    print(f"Done! Thresholded by manual thresholds {MANUAL_THRESHOLDS[:-1]} → {OUTPUT_FILE}")


if dryrun:
    os.remove(OUTPUT_FILE)
    print("Dryrun - deleted output file")
