"""cluster_regions.py — compute climate/drought features per region and
group regions with weighted Ward hierarchical clustering.

Outputs:
    region_features.csv      one row per region, raw feature values
    region_clusters.csv      region_id -> cluster_id mapping (1..K_final)
    region_dendrogram.png    Ward dendrogram, color-cut at TARGET_K

Pipeline:
    1. Restrict train.csv to last LOOKBACK_YEARS per region.
    2. Compute 14 features per region:
         - climate baseline (annual_mean_precip/temp, temp_range, aridity)
         - seasonality phase (sin/cos of month_of_max_temp & max_precip)
         - seasonality strength (CoV of monthly precip & temp)
         - drought behaviour (drought_rate, mean_score_when_drought)
         - precip→drought sensitivity (correlation, normalized threshold)
    3. Z-score, multiply by WEIGHTS, Ward-linkage at TARGET_K.
    4. Merge any cluster smaller than MIN_CLUSTER_SIZE into its nearest
       neighbour (otherwise too small to train a useful per-cluster model).
    5. Renumber clusters 1..K_final and save.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

DATA = Path('data')

# --- knobs -------------------------------------------------------------------
LOOKBACK_YEARS   = 10
TARGET_K         = 30
MIN_CLUSTER_SIZE = 30
# -----------------------------------------------------------------------------

WEIGHTS = {
    # hemisphere — the single most important separator
    # update: apparently there is only northern hemishpere, use low weights
    'sin_max_temp_month':       0.3,
    'cos_max_temp_month':       0.3,
    # transferability hint — drought response to precip
    'precip_drought_corr':      2.0,
    # drought regime
    'drought_rate':             1.5,
    'normalized_threshold':     1.5,
    # climate baseline
    'annual_mean_precip':       1.2,
    'annual_temp_range':        1.2,
    # wet-season phase + climate texture
    'sin_max_precip_month':     1.0,
    'cos_max_precip_month':     1.0,
    'precip_seasonality':       1.0,
    'annual_mean_temp':         1.0,
    # less distinctive
    'temp_seasonality':         0.7,
    'mean_score_when_drought':  0.7,
    'aridity_index':            0.5,
}
FEATURES = list(WEIGHTS.keys())

OUT_FEATURES   = Path('region_features.csv')
OUT_CLUSTERS   = Path('region_clusters.csv')
OUT_DENDROGRAM = Path('region_dendrogram.png')


def add_calendar(df):
    ymd = df['date'].str.split('-', expand=True).astype(int)
    df['year']    = ymd[0]
    df['month']   = ymd[1]
    df['day_idx'] = ymd[0] * 372 + (ymd[1] - 1) * 31 + (ymd[2] - 1)
    return df


def best_threshold(features, labels):
    """Per-region precip-91d cut that maximises drought-prediction accuracy."""
    n = len(features)
    if n == 0:
        return np.nan
    labels = labels.astype(int)
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == n:
        return np.nan
    order = np.argsort(features)
    f = features[order]
    l = labels[order]
    cum = np.concatenate(([0], np.cumsum(l)))
    correct = 2 * cum - np.arange(n + 1) + (n - n_pos)
    transitions = np.concatenate(([0], np.where(np.diff(f) > 0)[0] + 1, [n]))
    best_i = int(transitions[np.argmax(correct[transitions])])
    if best_i == 0:
        return float(f[0]) - 1.0
    if best_i == n:
        return float(f[-1]) + 1.0
    return float((f[best_i - 1] + f[best_i]) / 2)


def main():
    print(f'cluster_regions: lookback={LOOKBACK_YEARS}y, target_K={TARGET_K}, '
          f'min_cluster_size={MIN_CLUSTER_SIZE}')

    print('Loading train.csv...')
    train = pd.read_csv(DATA / 'train.csv',
                        usecols=['region_id', 'date', 'prec', 'tmp', 'score'])
    train = add_calendar(train)
    train = train.sort_values(['region_id', 'day_idx']).reset_index(drop=True)

    # Restrict to last LOOKBACK_YEARS per region (recent climate)
    rmax = train.groupby('region_id')['year'].max().rename('max_year')
    train = train.merge(rmax, left_on='region_id', right_index=True)
    train = train[train['year'] >= train['max_year'] - LOOKBACK_YEARS + 1].reset_index(drop=True)
    print(f'  {len(train):,} daily rows over {train["region_id"].nunique()} regions')

    print('Computing rolling precip_91d (for drought sensitivity features)...')
    train['precip_91d'] = (train.groupby('region_id', sort=False)['prec']
                                .rolling(91, min_periods=91).sum()
                                .reset_index(level=0, drop=True))

    # --- climate baseline (per region) --------------------------------------
    print('Computing climate baseline features...')
    region = train.groupby('region_id').agg(
        annual_mean_temp=('tmp', 'mean'),
        _daily_mean_precip=('prec', 'mean'),
    )
    region['annual_mean_precip'] = region['_daily_mean_precip'] * 365.25
    region.drop(columns=['_daily_mean_precip'], inplace=True)

    # --- monthly aggregates (one row per region-month) ----------------------
    monthly = train.groupby(['region_id', 'month']).agg(
        m_tmp=('tmp', 'mean'),
        m_prec=('prec', 'sum'),     # total precip across all years for that month
    ).reset_index()

    # annual_temp_range = peak_month_temp - trough_month_temp
    region['annual_temp_range'] = (monthly.groupby('region_id')['m_tmp']
                                          .agg(lambda s: s.max() - s.min()))

    # seasonality strength = CoV across months
    region['temp_seasonality'] = (monthly.groupby('region_id')['m_tmp']
                                          .agg(lambda s: s.std() / (abs(s.mean()) + 1e-6)))
    region['precip_seasonality'] = (monthly.groupby('region_id')['m_prec']
                                            .agg(lambda s: s.std() / (s.mean() + 1e-6)))

    # month_of_max_temp / max_precip, encoded as sin/cos for circularity
    max_temp_idx   = monthly.loc[monthly.groupby('region_id')['m_tmp'].idxmax()]
    max_precip_idx = monthly.loc[monthly.groupby('region_id')['m_prec'].idxmax()]
    max_t = max_temp_idx.set_index('region_id')['month']
    max_p = max_precip_idx.set_index('region_id')['month']
    region['sin_max_temp_month']   = np.sin(2 * np.pi * max_t / 12)
    region['cos_max_temp_month']   = np.cos(2 * np.pi * max_t / 12)
    region['sin_max_precip_month'] = np.sin(2 * np.pi * max_p / 12)
    region['cos_max_precip_month'] = np.cos(2 * np.pi * max_p / 12)

    # aridity = precip / (temp + 20) — offset keeps cold regions away from 0
    region['aridity_index'] = region['annual_mean_precip'] / (region['annual_mean_temp'] + 20.0)

    # --- drought behaviour (weekly score rows only) -------------------------
    print('Computing drought-behaviour features...')
    weekly = train[train['score'].notna() & train['precip_91d'].notna()].copy()
    weekly['drought'] = (weekly['score'] >= 1).astype(int)

    region['drought_rate'] = weekly.groupby('region_id')['drought'].mean()

    drought_only = weekly[weekly['drought'] == 1]
    region['mean_score_when_drought'] = (drought_only.groupby('region_id')['score']
                                                      .mean().fillna(1.0))

    # precip → drought correlation (Pearson; cheap, captures the sign + strength)
    print('Computing per-region precip-drought correlation...')
    def _corr(g):
        if len(g) < 5:
            return 0.0
        c = g['precip_91d'].corr(g['score'])
        return c if not pd.isna(c) else 0.0
    region['precip_drought_corr'] = weekly.groupby('region_id').apply(_corr)

    # normalized threshold = per-region precip-91d cut / region's mean precip-91d
    print('Computing per-region drought thresholds...')
    thresholds = {}
    region_mean_p91 = weekly.groupby('region_id')['precip_91d'].mean()
    for rid, g in weekly.groupby('region_id', sort=False):
        thresholds[rid] = best_threshold(g['precip_91d'].values, g['drought'].values)
    region['_threshold'] = pd.Series(thresholds)
    region['normalized_threshold'] = (region['_threshold'] /
                                       region_mean_p91.replace(0, np.nan))
    region.drop(columns=['_threshold'], inplace=True)

    # Fill any leftover NaNs with per-column medians (rare edge cases like
    # all-zero precip or all-no-drought history)
    n_nan_before = region[FEATURES].isna().sum().sum()
    if n_nan_before:
        print(f'  filling {int(n_nan_before)} NaN cells with per-column medians')
        region[FEATURES] = region[FEATURES].fillna(region[FEATURES].median())

    # Save raw features
    region[FEATURES].to_csv(OUT_FEATURES)
    print(f'\nWrote {OUT_FEATURES}  ({len(region)} regions, {len(FEATURES)} features)')

    # --- diagnostic: highly-correlated feature pairs ------------------------
    print('\nFeature pairs with |corr| > 0.7 (consider dropping/downweighting):')
    cm = region[FEATURES].corr()
    high = [(f1, f2, cm.loc[f1, f2])
            for i, f1 in enumerate(FEATURES)
            for f2 in FEATURES[i + 1:]
            if abs(cm.loc[f1, f2]) > 0.7]
    if high:
        for f1, f2, c in sorted(high, key=lambda x: -abs(x[2])):
            print(f'  {f1:25s} <-> {f2:25s}  {c:+.3f}')
    else:
        print('  (none — feature set is non-redundant at this threshold)')

    # --- standardise + weight + Ward ---------------------------------------
    print('\nStandardising + weighting + Ward linkage...')
    X = region[FEATURES].copy()
    X = (X - X.mean()) / (X.std() + 1e-9)
    w = np.array([WEIGHTS[f] for f in FEATURES])
    Xw = X.values * w
    Z = linkage(Xw, method='ward')

    # Dendrogram (color cut at TARGET_K)
    plt.figure(figsize=(20, 8))
    color_threshold = Z[-(TARGET_K - 1), 2] - 1e-9 if TARGET_K > 1 else 0
    dendrogram(Z, no_labels=True, color_threshold=color_threshold)
    plt.axhline(color_threshold, color='gray', ls='--', alpha=0.5)
    plt.title(f'Ward dendrogram on weighted features  (color cut at K={TARGET_K})')
    plt.ylabel('Linkage distance')
    plt.tight_layout()
    plt.savefig(OUT_DENDROGRAM, dpi=80)
    plt.close()
    print(f'Saved dendrogram to {OUT_DENDROGRAM}')

    # Initial clusters
    clusters = fcluster(Z, t=TARGET_K, criterion='maxclust')
    region['cluster'] = clusters
    sizes = region['cluster'].value_counts().sort_index()
    print(f'\nInitial cluster sizes (K={TARGET_K}):  '
          f'min={sizes.min()}  median={sizes.median():.0f}  '
          f'max={sizes.max()}  mean={sizes.mean():.1f}')
    print(f'  count < {MIN_CLUSTER_SIZE}: {(sizes < MIN_CLUSTER_SIZE).sum()}')

    # --- merge tiny clusters into nearest neighbour -------------------------
    centroids = (pd.DataFrame(Xw, index=region.index, columns=FEATURES)
                   .assign(cluster=clusters)
                   .groupby('cluster')[FEATURES].mean())
    tiny = sizes[sizes < MIN_CLUSTER_SIZE].index.tolist()
    if tiny:
        print(f'\nMerging {len(tiny)} tiny clusters into nearest:')
        for tc in tiny:
            others = centroids.drop(tiny)
            d = np.linalg.norm(others.values - centroids.loc[tc].values, axis=1)
            target = others.index[int(np.argmin(d))]
            region.loc[region['cluster'] == tc, 'cluster'] = target
            print(f'  cluster {tc:3d} (n={int(sizes[tc]):3d})  ->  cluster {target}')

    # Renumber 1..K_final
    remap = {old: new + 1 for new, old in enumerate(sorted(region['cluster'].unique()))}
    region['cluster'] = region['cluster'].map(remap)

    final_sizes = region['cluster'].value_counts().sort_index()
    print(f'\nFinal: {len(final_sizes)} clusters  '
          f'(sizes min={final_sizes.min()}  median={final_sizes.median():.0f}  '
          f'max={final_sizes.max()})')

    # --- per-cluster profile ------------------------------------------------
    profile = region.groupby('cluster')[FEATURES].mean().round(2)
    profile.insert(0, 'size', final_sizes)
    print('\nPer-cluster feature profile (raw, not z-scored):')
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(profile.to_string())

    # Save assignments
    region[['cluster']].reset_index().to_csv(OUT_CLUSTERS, index=False)
    print(f'\nWrote {OUT_CLUSTERS}  ({len(region)} regions)')


if __name__ == '__main__':
    main()
