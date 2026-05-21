"""Average drought score across all regions over a window defined by
days-since-each-region's-start.

Each region has ~5,480 daily train rows starting at its own date 0. The window
is the same number of days for every region, but maps to different calendar
windows per region. Years are converted with DAYS_PER_YEAR (365 by default).

Tweak START_YEARS / END_YEARS (None = open-ended on that side) and rerun.
"""
from pathlib import Path
import pandas as pd

DATA = Path('data')
DAYS_PER_YEAR = 365
START_YEARS = 5    # None = from each region's first row
END_YEARS   = 10    # None = to each region's last row


def avg_score(start_years=START_YEARS, end_years=END_YEARS,
              days_per_year=DAYS_PER_YEAR):
    print('Loading train.csv...')
    train = pd.read_csv(DATA / 'train.csv', usecols=['region_id', 'date', 'score'])

    # for start_years in range(15):
    #     end_years = start_years + 1
    if True:

        # Sort by region then date for a stable per-region day offset. Date strings
        # don't sort lexicographically across 4- vs 5-digit years, so build a
        # monotonic day_idx (same trick as region_exploration.ipynb).
        ymd = train['date'].str.split('-', expand=True).astype(int)
        train['day_idx'] = ymd[0] * 372 + (ymd[1] - 1) * 31 + (ymd[2] - 1)
        train = train.sort_values(['region_id', 'day_idx']).reset_index(drop=True)

        # Day offset since each region's first row (rows are daily).
        train['day_offset'] = train.groupby('region_id').cumcount()

        start = 0 if start_years is None else int(start_years * days_per_year)
        end   = train['day_offset'].max() + 1 if end_years is None else int(end_years * days_per_year)

        window = train[(train['day_offset'] >= start) & (train['day_offset'] < end)]
        window = window.dropna(subset=['score'])
        mean = float(window['score'].mean()) if len(window) else float('nan')

        print(f'Window: years [{start_years}, {end_years})  ->  days [{start}, {end})')
        print(f'Rows in window: {len(window):,}  '
            f'(across {window["region_id"].nunique()} regions)')
        print(f'Mean score:     {mean:.4f}')
    return mean


if __name__ == '__main__':
    avg_score()
