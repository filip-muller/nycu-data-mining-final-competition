"""Baseline submission: predict drought scores from regional monthly stats.

For each region:
  1. Take the last LOOKBACK_YEARS of weekly score observations from train.csv.
  2. Compute the mean (or QUANTILE) of score per calendar month.
  3. For each pred_week (1..5), figure out which month it lands in (the 5
     weeks immediately following that region's test window), and emit the
     corresponding monthly statistic.

Set QUANTILE to a number in [0, 1] (e.g. 0.6) to use that quantile instead
of the mean. Leave as None for the mean.

Calendar notes (mirrors region_exploration.ipynb): the dataset uses a
synthetic calendar with 4- and 5-digit years, so we use a monotonic ordinal
`day_idx = year*372 + (month-1)*31 + (day-1)` instead of real datetime. The
month of `day_idx + 7*k` is `((day_idx + 7*k) % 372) // 31 + 1`. This is off
by one day at month boundaries but lands in the correct month bucket.
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path('data')
LOOKBACK_YEARS = 10    # how many recent years of train history to average over
QUANTILE = 0.71       # None = use mean; otherwise e.g. 0.6
OUT = Path(
    f'submission_monthly_averages_{LOOKBACK_YEARS}.csv' if QUANTILE is None
    else f'submission_monthly_q{QUANTILE}_{LOOKBACK_YEARS}.csv'
)


def add_calendar(df):
    ymd = df['date'].str.split('-', expand=True).astype(int)
    df['year']    = ymd[0]
    df['month']   = ymd[1]
    df['day_idx'] = ymd[0] * 372 + (ymd[1] - 1) * 31 + (ymd[2] - 1)
    return df


def month_at(day_idx_value):
    return (int(day_idx_value) % 372) // 31 + 1


def main():
    stat = 'mean' if QUANTILE is None else f'q{QUANTILE}'
    print(f'Loading data (lookback={LOOKBACK_YEARS} years, stat={stat})...')
    train = pd.read_csv(DATA / 'train.csv', usecols=['region_id', 'date', 'score'])
    test  = pd.read_csv(DATA / 'test.csv',  usecols=['region_id', 'date'])
    sample = pd.read_csv(DATA / 'sample_submission.csv', usecols=['region_id'])

    print("Loaded data")

    train = add_calendar(train)
    test  = add_calendar(test)
    train = train[train['score'].notna()]  # weekly score observations only

    # Restrict each region's train data to the last LOOKBACK_YEARS years.
    region_max_year = train.groupby('region_id')['year'].max().rename('max_year')
    train = train.merge(region_max_year, left_on='region_id', right_index=True)
    train = train[train['year'] >= train['max_year'] - LOOKBACK_YEARS + 1]
    print(f'  {len(train):,} weekly train rows after {LOOKBACK_YEARS}-yr filter')

    # Statistic per (region, month), per region, and globally.
    if QUANTILE is None:
        monthly_s      = train.groupby(['region_id', 'month'])['score'].mean()
        region_overall = train.groupby('region_id')['score'].mean()
        global_val     = float(train['score'].mean())
    else:
        monthly_s      = train.groupby(['region_id', 'month'])['score'].quantile(QUANTILE)
        region_overall = train.groupby('region_id')['score'].quantile(QUANTILE)
        global_val     = float(train['score'].quantile(QUANTILE))
    monthly = monthly_s.unstack('month')

    # Day_idx of the last row of each region's test window.
    test_last_idx = test.groupby('region_id')['day_idx'].max()

    rows = []
    for region in sample['region_id']:
        last_idx = test_last_idx.get(region)
        if last_idx is None:
            rows.append([region, global_val, global_val, global_val,
                         global_val, global_val])
            continue

        target_months = [month_at(last_idx + 7 * k) for k in range(1, 6)]
        fallback = float(region_overall.get(region, global_val))

        if region in monthly.index:
            region_row = monthly.loc[region]
            preds = [region_row.get(m, np.nan) for m in target_months]
            preds = [fallback if pd.isna(v) else float(v) for v in preds]
        else:
            preds = [fallback] * 5
        rows.append([region, *preds])

    sub = pd.DataFrame(
        rows,
        columns=['region_id', 'pred_week1', 'pred_week2', 'pred_week3',
                 'pred_week4', 'pred_week5'],
    )
    sub.to_csv(OUT, index=False)
    print(f'Wrote {OUT}  ({len(sub)} regions)')


if __name__ == '__main__':
    main()
