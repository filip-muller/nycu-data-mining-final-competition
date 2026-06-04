"""deep_drought.py — Transformer + static-context drought predictor.

ONE-FILE script. Subcommands:
    python deep_drought.py train [--cfg key=value ...]
    python deep_drought.py predict --checkpoint PATH [--output PATH]

What it does:
- Loads train.csv / test.csv (+ optional region_clusters.csv).
- Builds two parallel input streams per training sample:
    1) a 91-day daily meteorology window (Transformer-encoded)
    2) a ~25-scalar long-context vector of region- and cluster-level
       statistics plus multi-window rolling aggregates
   …plus region / cluster / week-index embeddings.
- Predicts 5 future weekly scores via an ordinal head (P(score>=k) for
  k=1..5) per week. Loss = weighted (BCE + MAE-on-expected-score).
- Trains with sliding-window subsampling, within-cluster mixup, label
  noise, curriculum on horizon, embedding dropout. Saves checkpoints.
- At inference: TTA via window jittering + soft ensembling across saved
  checkpoints if you want; writes a Kaggle-format submission CSV.

Almost everything is gated by Config flags so you can ablate quickly.

Notes on inputs the script depends on:
    data/train.csv         required
    data/test.csv          required
    data/sample_submission.csv  required (just for region ordering)
    region_clusters.csv    optional (if missing or cluster embedding disabled,
                                     cluster features are zeroed out)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict, fields, replace
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    # --- paths -------------------------------------------------------------
    data_dir: str = 'data'
    cluster_csv: str = 'region_clusters.csv'        # ignored if missing or use_cluster_embedding=False
    checkpoint_dir: str = 'checkpoints'
    submission_path: str = 'submissions/submission_deep_drought.csv'
    run_name: str = field(default_factory=lambda: time.strftime('run_%Y%m%d_%H%M%S'))

    # --- task --------------------------------------------------------------
    drought_threshold: float = 1.0
    score_max: int = 5
    n_pred_weeks: int = 5
    replicate_prediction: bool = False              # if True: ONE shared prediction for all 5 weeks (no per-week
                                                    # target-month or week-index embedding; single head forward
                                                    # broadcast to [B, 5, n_out]; loss averages over the 5 targets;
                                                    # submission writes the same value 5 times).

    # --- data --------------------------------------------------------------
    lookback_years: int = 10                        # of train history per region
    window_days: int = 91                           # = test length
    val_holdout_year: bool = True                   # hold out most-recent train years per region
    val_holdout_years: int = 1                      # number of recent years held out (if val_holdout_year=True)
    samples_per_region_per_epoch: int = 200         # subsample sliding windows per region

    # --- distribution-shift handling --------------------------------------
    # The competition test set has a different score distribution than the
    # training data (see notes.md). These two flags let you correct for that.
    test_target_distribution: Tuple[float, ...] = (0.35, 0.275, 0.24, 0.09, 0.04, 0.005)
    use_test_dist_weighted_val: bool = True         # also report an importance-weighted val MAE (matches test class distribution); used as 'best' criterion
    use_test_dist_sample_weights: bool = False      # multiply each train sample's loss by P_test(score)/P_train(score)
    sample_weight_clip: float = 10.0                # cap on score-weight ratios (both directions, e.g. 0.1 - 10)

    # --- extra engineered features (additions to long_ctx) ----------------
    # Master flag turns ALL of them on. Each individual flag can override:
    #   None    -> inherit from extra_features (default)
    #   True    -> force on
    #   False   -> force off (use --cfg feat_xxx=none to reset to inherit)
    extra_features: bool = False
    feat_vpd:                 Optional[bool] = None  # vpd_30d = temp_30d - dp_tmp_30d (vapor pressure deficit proxy)
    feat_dry_streak:          Optional[bool] = None  # current_dry_streak (consecutive days with prec<1mm ending at window end)
    feat_wet_days_30d:        Optional[bool] = None  # count of days with prec>=1mm in last 30 days
    feat_precip_trend:        Optional[bool] = None  # precip_30d - (precip_91d - precip_30d) / 2  (recent minus older)
    feat_precip_z_for_month:  Optional[bool] = None  # z-score precip_91d against this region's same-month history

    # daily columns (must all exist in train.csv and test.csv)
    daily_cols: Tuple[str, ...] = (
        'prec', 'tmp', 'tmp_min', 'tmp_max', 'humidity',
        'dp_tmp', 'wb_tmp', 'wind', 'surf_tmp', 'surf_pre',
    )

    # --- long-context vector (precomputed per sample) ----------------------
    use_long_context: bool = True
    include_rolling_aggs: bool = True               # precip/temp/humidity rolling at window end
    include_region_stats: bool = True               # drought rate, mean score, etc.
    include_thresholds:   bool = True               # region & per-(region, month) precip thresholds
    include_cluster_aggs: bool = True               # cluster drought rate / monthly threshold (needs cluster_csv)

    # --- identity & per-week embeddings -----------------------------------
    # All four embeddings below feed into the fusion / head stages, not the
    # Transformer. They're cheap (small dims × small vocab sizes) so feel
    # free to grow them, but the model isn't very sensitive past 16-32.

    use_region_embedding: bool = True
    region_emb_dim: int = 16                        # one 16-d vector per region (2248 entries)
    region_emb_init_from_cluster: bool = True       # init each region to its cluster centroid (regularizer)
    region_emb_dropout: float = 0.2                 # per-sample chance of zeroing the region vector during training

    use_cluster_embedding: bool = True
    cluster_emb_dim: int = 8                        # one 8-d vector per cluster (~22 entries)

    # Per-week conditioning: each future week gets month + week-index embeddings
    # appended to the fused representation before the (shared) prediction head.
    use_target_month_embedding: bool = True
    month_emb_dim: int = 8                          # one 8-d vector per calendar month (12 + 1 entries)
    use_week_index_embedding: bool = True
    week_emb_dim: int = 16                          # one 16-d vector per future-week slot (5 entries)

    # --- model -------------------------------------------------------------
    use_daily_branch: bool = True                   # if False: skip Transformer + daily input entirely; MLP baseline on long_ctx + embeddings
    freeze_daily_branch: bool = False               # if True: build the daily Transformer but never update its weights
                                                    # (acts as a stochastic noise-injection / regularizer; the rest of the model trains normally)
    freeze_daily_branch_after_epoch: int = 0        # 0 = never; N > 0 = stop updating Transformer weights starting epoch N+1
                                                    # (train for N epochs normally, then freeze; lets the Transformer learn something cheap then lock it in)

    # `d_model` is the SHARED hidden width used by:
    #   - the Transformer encoder layers (its "d_model" hyperparameter)
    #   - the daily linear projection (F -> d_model) that feeds the Transformer
    #   - the long_ctx_mlp's output (so it can be cross-attended with daily)
    #   - the cross-attention fusion block
    # It's the "main backbone width". Set it bigger when scaling up.
    d_model: int = 96

    # Transformer-only knobs (ignored if use_daily_branch=false):
    n_transformer_layers: int = 3                   # number of stacked encoder layers
    n_heads: int = 4                                # attention heads per layer (must divide d_model)
    ffn_dim: int = 256                              # feed-forward inner dim inside each layer

    dropout: float = 0.1                            # applied throughout (Transformer, MLPs, head)
    use_attn_pooling: bool = True                   # daily-only: learned-query pool after Transformer (vs mean pool)
    use_cross_attn_fusion: bool = True              # daily-only: cross-attend daily on long-context (ignored if no daily branch)

    # `head_hidden_dim` is the hidden width of:
    #   - the fusion MLP (combines daily / long_ctx / identity -> single vector)
    #   - the per-week prediction head (shared MLP that produces ordinal logits)
    # Doesn't need to equal d_model; tune separately.
    head_hidden_dim: int = 128

    # --- output / loss -----------------------------------------------------
    use_ordinal_head: bool = True                   # 5 cumulative logits per week; expected = sum of sigmoids
    bce_weight: float = 1.0
    mae_weight: float = 0.3

    # --- training ----------------------------------------------------------
    n_epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    warmup_steps: int = 500
    seed: int = 42
    device: str = 'cuda'                            # falls back to cpu if cuda unavailable
    num_workers: int = 4
    pin_memory: bool = True

    # --- augmentation / tricks --------------------------------------------
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    mixup_within_cluster_only: bool = True
    use_label_noise: bool = True
    label_noise_std: float = 0.1
    use_curriculum: bool = True
    curriculum_epochs: int = 3                      # epochs to train week-1 only before expanding

    # --- inference --------------------------------------------------------
    use_tta: bool = True
    n_tta_augments: int = 4

    # --- checkpointing ----------------------------------------------------
    save_every_n_epochs: int = 1                    # 1 = every epoch (set higher to thin out)
    save_best: bool = True

    # --- inference post-processing ----------------------------------------
    prediction_scale: float = 1.0                   # multiplier applied to final predictions only
                                                    # (try 0.85-0.95 to push toward conditional median)

    # --- logging ----------------------------------------------------------
    print_every_n_steps: int = 0                    # 0 = disable per-step logs (only per-epoch summary)


def cfg_from_cli_overrides(cfg: Config, overrides: List[str]) -> Config:
    """Apply --cfg key=value (or key:value) overrides to a Config."""
    if not overrides:
        return cfg
    kv = {}
    for o in overrides:
        for sep in ('=', ':'):
            if sep in o:
                k, v = o.split(sep, 1)
                kv[k.strip()] = v.strip()
                break
    type_map = {f.name: f.type for f in fields(cfg)}
    new = {}
    for k, v in kv.items():
        if k not in type_map:
            raise SystemExit(f'unknown config key: {k}')
        t = type_map[k]
        t_str = str(t)
        v_lower = v.lower()
        # Allow "none"/"inherit" to reset Optional fields to None
        if v_lower in ('none', 'null', 'inherit'):
            new[k] = None
        elif t is bool or t == 'bool' or 'bool' in t_str:
            new[k] = v_lower in ('1', 'true', 't', 'yes', 'y')
        elif t is int or t == 'int' or 'int' in t_str:
            new[k] = int(v)
        elif t is float or t == 'float' or 'float' in t_str:
            new[k] = float(v)
        else:
            new[k] = v
    return replace(cfg, **new)


# =============================================================================
# Calendar / threshold helpers (mirrors prior scripts)
# =============================================================================

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    ymd = df['date'].str.split('-', expand=True).astype(int)
    df['year']    = ymd[0]
    df['month']   = ymd[1]
    df['day_idx'] = ymd[0] * 372 + (ymd[1] - 1) * 31 + (ymd[2] - 1)
    return df


def best_threshold(features: np.ndarray, labels: np.ndarray) -> float:
    n = len(features)
    if n == 0:
        return float('nan')
    labels = labels.astype(int)
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == n:
        return float('nan')
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


def month_in_bucket(m: int, center: int) -> bool:
    d = abs(int(m) - int(center))
    return min(d, 12 - d) <= 1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Data preparation
# =============================================================================

@dataclass
class FeatureBundle:
    """Everything needed at training and inference, save-friendly."""
    daily_cols: List[str]
    long_ctx_cols: List[str]                                   # in order
    daily_means: np.ndarray                                    # [F] for standardization
    daily_stds: np.ndarray                                     # [F]
    long_ctx_means: np.ndarray                                 # [S]
    long_ctx_stds: np.ndarray                                  # [S]
    region_to_idx: Dict[str, int]                              # region_id -> 0..R-1
    region_to_cluster: Dict[str, int]                          # region_id -> 1..K (or 0 if no clusters)
    n_regions: int
    n_clusters: int
    # per-region static info needed at inference
    region_static: Dict[str, Dict]                             # region_id -> {feature_name: value}
    # per-(region, month-bucket) thresholds
    region_threshold_by_month: Dict[Tuple[str, int], float]    # (region_id, month) -> threshold
    cluster_threshold_by_month: Dict[Tuple[int, int], float]   # (cluster_id, month) -> threshold
    # per-(region, month) precip-91d distribution stats (mean, std) for the
    # feat_precip_z_for_month feature
    region_month_precip_stats: Dict[Tuple[str, int], Tuple[float, float]] = field(default_factory=dict)


def load_raw_data(cfg: Config):
    data_dir = Path(cfg.data_dir)
    daily_cols = list(cfg.daily_cols)
    train = pd.read_csv(data_dir / 'train.csv',
                        usecols=['region_id', 'date'] + daily_cols + ['score'])
    test = pd.read_csv(data_dir / 'test.csv',
                       usecols=['region_id', 'date'] + daily_cols)
    sample = pd.read_csv(data_dir / 'sample_submission.csv', usecols=['region_id'])
    train = add_calendar(train).sort_values(['region_id', 'day_idx']).reset_index(drop=True)
    test  = add_calendar(test).sort_values(['region_id', 'day_idx']).reset_index(drop=True)
    return train, test, sample


def load_clusters(cfg: Config) -> Tuple[Dict[str, int], int]:
    path = Path(cfg.cluster_csv)
    if not cfg.use_cluster_embedding and not cfg.include_cluster_aggs:
        return {}, 0
    if not path.exists():
        print(f'  (no {path} found; cluster features disabled)')
        return {}, 0
    df = pd.read_csv(path)
    mapping = dict(zip(df['region_id'], df['cluster'].astype(int)))
    K = int(df['cluster'].max())
    return mapping, K


def build_feature_bundle(cfg: Config,
                         train: pd.DataFrame,
                         test: pd.DataFrame,
                         region_to_cluster: Dict[str, int],
                         n_clusters: int) -> FeatureBundle:
    """Compute all reusable feature stats from train data only (no leakage)."""
    daily_cols = list(cfg.daily_cols)

    # last LOOKBACK_YEARS per region for stats
    print('Building feature bundle...')
    rmax = train.groupby('region_id')['year'].max().rename('max_year')
    tr = train.merge(rmax, left_on='region_id', right_index=True)
    tr = tr[tr['year'] >= tr['max_year'] - cfg.lookback_years + 1].reset_index(drop=True)

    # weekly score rows for drought-target stats
    has_score = tr['score'].notna()
    weekly = tr.loc[has_score].copy()
    weekly['drought'] = (weekly['score'] >= cfg.drought_threshold).astype(int)

    # if validating: exclude held-out year from stats
    if cfg.val_holdout_year:
        weekly_stats = weekly[weekly['year'] < weekly['max_year']].copy()
    else:
        weekly_stats = weekly

    # rolling features on the daily train data (used both as long-context inputs
    # at training time AND to compute z-scores against region history)
    print('  computing rolling features on train...')
    g = tr.groupby('region_id', sort=False)
    tr['precip_91d'] = g['prec'].rolling(91, min_periods=91).sum().reset_index(level=0, drop=True)
    tr['precip_30d'] = g['prec'].rolling(30, min_periods=30).sum().reset_index(level=0, drop=True)
    tr['precip_7d']  = g['prec'].rolling( 7, min_periods= 7).sum().reset_index(level=0, drop=True)
    tr['temp_30d']   = g['tmp'].rolling(30,  min_periods=30).mean().reset_index(level=0, drop=True)
    tr['temp_7d']    = g['tmp'].rolling( 7,  min_periods= 7).mean().reset_index(level=0, drop=True)
    tr['humidity_30d'] = g['humidity'].rolling(30, min_periods=30).mean().reset_index(level=0, drop=True)
    tr['dp_tmp_30d']   = g['dp_tmp'].rolling(30, min_periods=30).mean().reset_index(level=0, drop=True)
    tr['wind_30d']     = g['wind'].rolling(30, min_periods=30).mean().reset_index(level=0, drop=True)
    tr['surf_tmp_30d'] = g['surf_tmp'].rolling(30, min_periods=30).mean().reset_index(level=0, drop=True)
    tr['surf_pre_30d'] = g['surf_pre'].rolling(30, min_periods=30).mean().reset_index(level=0, drop=True)

    # per-region typical values (for z-scores)
    print('  computing per-region typical precip/temp distributions...')
    region_typical = tr.groupby('region_id').agg(
        region_p91_mean=('precip_91d', 'mean'),
        region_p91_std=('precip_91d', 'std'),
        annual_mean_precip=('prec', 'mean'),
        annual_mean_temp=('tmp', 'mean'),
    )
    region_typical['annual_mean_precip'] = region_typical['annual_mean_precip'] * 365.25
    # temp range from monthly means
    monthly_tmp = tr.groupby(['region_id', 'month'])['tmp'].mean().reset_index()
    region_typical['annual_temp_range'] = (monthly_tmp.groupby('region_id')['tmp']
                                            .agg(lambda s: s.max() - s.min()))

    # per-region drought rate / score stats from weekly_stats
    print('  computing per-region drought stats...')
    weekly_stats_wf = weekly_stats.merge(
        tr[['region_id', 'day_idx', 'precip_91d', 'temp_30d']],
        on=['region_id', 'day_idx'], how='left',
    ).dropna(subset=['precip_91d'])
    region_drought = weekly_stats_wf.groupby('region_id').agg(
        region_drought_rate=('drought', 'mean'),
        region_mean_score=('score', 'mean'),
    )
    drought_only = weekly_stats_wf[weekly_stats_wf['drought'] == 1]
    region_drought['region_mean_score_when_drought'] = (
        drought_only.groupby('region_id')['score'].mean().fillna(1.0))

    # per-region precip-91d threshold (global)
    print('  computing per-region precip-91d thresholds...')
    region_thr = {}
    for region, gdf in weekly_stats_wf.groupby('region_id', sort=False):
        region_thr[region] = best_threshold(gdf['precip_91d'].values, gdf['drought'].values)

    # per-(region, month) threshold (with ±1-month bucket)
    print('  computing per-(region, month) thresholds...')
    region_thr_month: Dict[Tuple[str, int], float] = {}
    for region, gdf in weekly_stats_wf.groupby('region_id', sort=False):
        months = gdf['month'].values
        p91 = gdf['precip_91d'].values
        d = gdf['drought'].values
        for m_c in range(1, 13):
            bucket = {((m_c - 2) % 12) + 1, m_c, (m_c % 12) + 1}
            mask = np.isin(months, list(bucket))
            t = best_threshold(p91[mask], d[mask]) if mask.sum() >= 30 else np.nan
            region_thr_month[(region, m_c)] = t

    # per-(region, month) precip-91d distribution stats — for feat_precip_z_for_month
    region_month_p91_stats: Dict[Tuple[str, int], Tuple[float, float]] = {}
    if _resolve_feat(cfg.feat_precip_z_for_month, cfg.extra_features):
        print('  computing per-(region, month) precip-91d stats...')
        grouped = weekly_stats_wf.groupby(['region_id', 'month'])['precip_91d']
        for (region, m), gdf in grouped:
            vals = gdf.values
            if len(vals) >= 2:
                region_month_p91_stats[(region, int(m))] = (float(vals.mean()), float(vals.std()))
            elif len(vals) == 1:
                region_month_p91_stats[(region, int(m))] = (float(vals[0]), 0.0)

    # cluster-level threshold per (cluster, month)
    cluster_thr_month: Dict[Tuple[int, int], float] = {}
    cluster_stats: Dict[int, Dict[str, float]] = {}
    if cfg.include_cluster_aggs and region_to_cluster:
        print('  computing per-(cluster, month) thresholds + cluster stats...')
        ws = weekly_stats_wf.copy()
        ws['cluster'] = ws['region_id'].map(region_to_cluster)
        for cluster_id, gdf in ws.groupby('cluster', sort=False):
            cluster_stats[int(cluster_id)] = {
                'cluster_drought_rate': float(gdf['drought'].mean()),
            }
            months = gdf['month'].values
            p91 = gdf['precip_91d'].values
            d = gdf['drought'].values
            for m_c in range(1, 13):
                bucket = {((m_c - 2) % 12) + 1, m_c, (m_c % 12) + 1}
                mask = np.isin(months, list(bucket))
                t = best_threshold(p91[mask], d[mask]) if mask.sum() >= 50 else np.nan
                cluster_thr_month[(int(cluster_id), m_c)] = t

    # consolidate per-region static features
    region_static: Dict[str, Dict] = {}
    for region in tr['region_id'].unique():
        rs = {}
        if region in region_typical.index:
            rs.update(region_typical.loc[region].to_dict())
        if region in region_drought.index:
            rs.update(region_drought.loc[region].to_dict())
        rs['region_threshold'] = region_thr.get(region, np.nan)
        if region_to_cluster:
            c = region_to_cluster.get(region, 0)
            cs = cluster_stats.get(c, {'cluster_drought_rate': np.nan})
            rs['cluster_drought_rate'] = cs['cluster_drought_rate']
        else:
            rs['cluster_drought_rate'] = np.nan
        region_static[region] = rs

    # region index
    regions_sorted = sorted(set(tr['region_id'].unique()) | set(test['region_id'].unique()))
    region_to_idx = {r: i for i, r in enumerate(regions_sorted)}

    # standardize daily columns (global stats)
    print('  computing daily standardization stats...')
    daily_means = tr[daily_cols].mean().values.astype(np.float32)
    daily_stds  = tr[daily_cols].std().replace(0, 1).values.astype(np.float32)

    # decide long-context column order
    long_ctx_cols = _build_long_ctx_column_list(cfg)
    # for standardization stats on long-context, we need a quick pass over training samples
    # (medians work fine; we'll just use 0/1 for stats since most are derived). We compute
    # using a sample of training rows below.
    long_ctx_means, long_ctx_stds = _compute_long_ctx_stats(
        cfg, tr, region_to_cluster, region_static, region_thr_month, cluster_thr_month,
        long_ctx_cols, daily_means, daily_stds,
    )

    return FeatureBundle(
        daily_cols=daily_cols,
        long_ctx_cols=long_ctx_cols,
        daily_means=daily_means,
        daily_stds=daily_stds,
        long_ctx_means=long_ctx_means,
        long_ctx_stds=long_ctx_stds,
        region_to_idx=region_to_idx,
        region_to_cluster=region_to_cluster,
        n_regions=len(regions_sorted),
        n_clusters=n_clusters,
        region_static=region_static,
        region_threshold_by_month=region_thr_month,
        cluster_threshold_by_month=cluster_thr_month,
        region_month_precip_stats=region_month_p91_stats,
    )


def _resolve_feat(individual: Optional[bool], master: bool) -> bool:
    """Optional[bool] flag with a master fallback. None = inherit."""
    return master if individual is None else bool(individual)


def _build_long_ctx_column_list(cfg: Config) -> List[str]:
    cols: List[str] = []
    if not cfg.use_long_context:
        return cols
    if cfg.include_rolling_aggs:
        cols += ['precip_91d', 'precip_30d', 'precip_7d',
                 'temp_30d', 'temp_7d',
                 'humidity_30d', 'dp_tmp_30d', 'wind_30d',
                 'surf_tmp_30d', 'surf_pre_30d']
    if cfg.include_region_stats:
        cols += ['region_drought_rate', 'region_mean_score',
                 'region_mean_score_when_drought',
                 'annual_mean_precip', 'annual_mean_temp', 'annual_temp_range',
                 'region_p91_mean']
    if cfg.include_thresholds:
        cols += ['region_threshold',
                 'precip_91d_vs_region_threshold',
                 'region_threshold_month',
                 'precip_91d_vs_region_threshold_month',
                 'precip_91d_z_in_region']
    if cfg.include_cluster_aggs:
        cols += ['cluster_drought_rate',
                 'cluster_threshold_month',
                 'precip_91d_vs_cluster_threshold_month']
    # --- extra engineered features (gated by config) ---
    if _resolve_feat(cfg.feat_vpd, cfg.extra_features):
        cols += ['vpd_30d']
    if _resolve_feat(cfg.feat_dry_streak, cfg.extra_features):
        cols += ['current_dry_streak']
    if _resolve_feat(cfg.feat_wet_days_30d, cfg.extra_features):
        cols += ['wet_days_30d']
    if _resolve_feat(cfg.feat_precip_trend, cfg.extra_features):
        cols += ['precip_recent_minus_old']
    if _resolve_feat(cfg.feat_precip_z_for_month, cfg.extra_features):
        cols += ['precip_91d_z_in_region_for_month']
    return cols


def _compute_long_ctx_stats(cfg, tr, region_to_cluster, region_static,
                             region_thr_month, cluster_thr_month,
                             long_ctx_cols, daily_means, daily_stds):
    """Compute per-feature mean/std over a sample of training rows."""
    S = len(long_ctx_cols)
    if S == 0:
        return np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)
    # Sample weekly rows for fitting standardization
    has_score = tr['score'].notna() & tr['precip_91d'].notna()
    rows = tr[has_score]
    if len(rows) > 200_000:
        rows = rows.sample(n=200_000, random_state=0)
    feats = np.zeros((len(rows), S), dtype=np.float32)
    region_ids = rows['region_id'].values
    months = rows['month'].values
    precip_91d = rows['precip_91d'].values
    rolling_lookup = {
        c: rows[c].values for c in
        ['precip_91d', 'precip_30d', 'precip_7d', 'temp_30d', 'temp_7d',
         'humidity_30d', 'dp_tmp_30d', 'wind_30d', 'surf_tmp_30d', 'surf_pre_30d']
        if c in rows.columns
    }
    # NOTE: vpd / dry-streak / wet-days / precip-trend / per-month-z aren't
    # stored as rolling columns on `rows`, but we can derive them here for
    # standardization-fit purposes.
    for i, col in enumerate(long_ctx_cols):
        if col in rolling_lookup:
            feats[:, i] = rolling_lookup[col]
        elif col == 'vpd_30d':
            t30 = rows['temp_30d'].values if 'temp_30d' in rows.columns else 0
            d30 = rows['dp_tmp_30d'].values if 'dp_tmp_30d' in rows.columns else 0
            feats[:, i] = t30 - d30
        elif col == 'precip_recent_minus_old':
            p30 = rows['precip_30d'].values
            p91 = rows['precip_91d'].values
            feats[:, i] = p30 - (p91 - p30) / 2.0
        elif col == 'precip_91d_z_in_region_for_month':
            # standardization-fit only sees a sampled subset; populate with
            # raw differences then standardize.
            p91 = rows['precip_91d'].values
            r_arr = region_ids
            m_arr = rows['month'].values
            # Look up per-(region, month) stats; missing -> 0
            # (we don't have access here since this function predates the
            # bundle. just use 0 as placeholder — fine for standardization fit.)
            feats[:, i] = 0.0
        elif col == 'current_dry_streak' or col == 'wet_days_30d':
            # rolling columns present on `rows`
            if col in rows.columns:
                feats[:, i] = rows[col].values
            else:
                feats[:, i] = 0.0
        elif col.startswith('region_threshold_month'):
            feats[:, i] = np.array([region_thr_month.get((r, m), np.nan)
                                     for r, m in zip(region_ids, months)])
        elif col == 'precip_91d_vs_region_threshold':
            t = np.array([region_static[r].get('region_threshold', np.nan) for r in region_ids])
            feats[:, i] = precip_91d - t
        elif col == 'precip_91d_vs_region_threshold_month':
            t = np.array([region_thr_month.get((r, m), np.nan)
                          for r, m in zip(region_ids, months)])
            feats[:, i] = precip_91d - t
        elif col == 'precip_91d_z_in_region':
            mean = np.array([region_static[r].get('region_p91_mean', np.nan) for r in region_ids])
            std = np.array([region_static[r].get('region_p91_std', np.nan) for r in region_ids])
            feats[:, i] = (precip_91d - mean) / np.where((std == 0) | np.isnan(std), 1.0, std)
        elif col.startswith('cluster_threshold_month'):
            feats[:, i] = np.array([
                cluster_thr_month.get((region_to_cluster.get(r, 0), m), np.nan)
                for r, m in zip(region_ids, months)])
        elif col == 'precip_91d_vs_cluster_threshold_month':
            t = np.array([cluster_thr_month.get((region_to_cluster.get(r, 0), m), np.nan)
                          for r, m in zip(region_ids, months)])
            feats[:, i] = precip_91d - t
        elif col in ('region_drought_rate', 'region_mean_score',
                     'region_mean_score_when_drought',
                     'annual_mean_precip', 'annual_mean_temp', 'annual_temp_range',
                     'region_p91_mean', 'region_threshold',
                     'cluster_drought_rate'):
            feats[:, i] = np.array([region_static[r].get(col, np.nan) for r in region_ids])
        else:
            # safety net
            feats[:, i] = 0.0
    means = np.nanmean(feats, axis=0).astype(np.float32)
    stds = np.nanstd(feats, axis=0).astype(np.float32)
    stds[stds == 0] = 1.0
    return means, stds


# =============================================================================
# Per-region daily arrays + sample list
# =============================================================================

@dataclass
class RegionDailyArrays:
    """Per-region contiguous (n_days, F) array of standardised daily features, plus
    the same-length rolling feature arrays for quick window-end lookup."""
    daily: Dict[str, np.ndarray]            # region_id -> (n_days, F) float32
    rolling: Dict[str, Dict[str, np.ndarray]]   # region_id -> col -> (n_days,) float32
    day_idx_arr: Dict[str, np.ndarray]      # region_id -> (n_days,) int64
    weekly_rows: Dict[str, np.ndarray]      # region_id -> indices into daily/rolling that have a score
    score_arr: Dict[str, np.ndarray]        # region_id -> (n_days,) float32 (NaN where no score)


def build_region_arrays(train: pd.DataFrame,
                        bundle: FeatureBundle,
                        cfg: Config) -> RegionDailyArrays:
    """Slice the train DataFrame into per-region numpy arrays for fast sampling."""
    print('Building per-region arrays...')

    # Rolling features needed for long-context at window end
    rolling_cols = ['precip_91d', 'precip_30d', 'precip_7d', 'temp_30d', 'temp_7d',
                    'humidity_30d', 'dp_tmp_30d', 'wind_30d', 'surf_tmp_30d', 'surf_pre_30d']
    train = train.copy()
    g = train.groupby('region_id', sort=False)
    train['precip_91d'] = g['prec'].rolling(91, min_periods=91).sum().reset_index(level=0, drop=True)
    train['precip_30d'] = g['prec'].rolling(30, min_periods=30).sum().reset_index(level=0, drop=True)
    train['precip_7d']  = g['prec'].rolling( 7, min_periods= 7).sum().reset_index(level=0, drop=True)
    train['temp_30d']   = g['tmp'].rolling(30,  min_periods=30).mean().reset_index(level=0, drop=True)
    train['temp_7d']    = g['tmp'].rolling( 7,  min_periods= 7).mean().reset_index(level=0, drop=True)
    train['humidity_30d'] = g['humidity'].rolling(30, min_periods=30).mean().reset_index(level=0, drop=True)
    train['dp_tmp_30d']   = g['dp_tmp'].rolling(30, min_periods=30).mean().reset_index(level=0, drop=True)
    train['wind_30d']     = g['wind'].rolling(30, min_periods=30).mean().reset_index(level=0, drop=True)
    train['surf_tmp_30d'] = g['surf_tmp'].rolling(30, min_periods=30).mean().reset_index(level=0, drop=True)
    train['surf_pre_30d'] = g['surf_pre'].rolling(30, min_periods=30).mean().reset_index(level=0, drop=True)

    # Extra rolling features for the optional engineered columns. They're cheap
    # so we always compute them; they're only USED if the cfg flag is on.
    # current_dry_streak: consecutive days with prec < 1mm ending at each day
    dry = (train['prec'] < 1.0).astype(int)
    # Group runs of identical dry/wet by detecting transitions, then cumcount within run.
    # `boundary` increments at each wet->dry or dry->wet transition. Pre-reset by groupby('region_id')
    # so streaks reset at region boundaries even when the previous region ended dry.
    boundary = (dry != dry.groupby(train['region_id']).shift()).astype(int)
    # Per-region group id: boundary cumsum, then groupby region to keep ids local
    train['_streak_group'] = boundary.groupby(train['region_id']).cumsum()
    streak_len = train.groupby(['region_id', '_streak_group']).cumcount() + 1
    train['current_dry_streak'] = (streak_len * dry).astype(np.float32)
    train.drop(columns=['_streak_group'], inplace=True)
    rolling_cols.append('current_dry_streak')

    # wet_days_30d: count of days with prec >= 1mm in last 30 days
    wet_ind = (train['prec'] >= 1.0).astype(float)
    train['wet_days_30d'] = (wet_ind.groupby(train['region_id'])
                              .rolling(30, min_periods=30).sum()
                              .reset_index(level=0, drop=True)).astype(np.float32)
    rolling_cols.append('wet_days_30d')

    # standardize daily features in place
    daily_cols = bundle.daily_cols
    train[daily_cols] = (train[daily_cols].values - bundle.daily_means) / bundle.daily_stds

    daily_arrays: Dict[str, np.ndarray] = {}
    rolling_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    day_arrays: Dict[str, np.ndarray] = {}
    weekly_rows: Dict[str, np.ndarray] = {}
    score_arrays: Dict[str, np.ndarray] = {}

    for region, gdf in train.groupby('region_id', sort=False):
        gdf = gdf.sort_values('day_idx').reset_index(drop=True)
        daily_arrays[region] = gdf[daily_cols].values.astype(np.float32)
        rolling_arrays[region] = {c: gdf[c].values.astype(np.float32) for c in rolling_cols}
        day_arrays[region] = gdf['day_idx'].values.astype(np.int64)
        weekly_mask = gdf['score'].notna().values
        weekly_rows[region] = np.where(weekly_mask)[0].astype(np.int64)
        score_arrays[region] = gdf['score'].values.astype(np.float32)

    return RegionDailyArrays(
        daily=daily_arrays,
        rolling=rolling_arrays,
        day_idx_arr=day_arrays,
        weekly_rows=weekly_rows,
        score_arr=score_arrays,
    )


# =============================================================================
# Sample generation
# =============================================================================

@dataclass
class SampleSpec:
    region_id: str
    end_row: int                  # row index in region's daily array (= last day of window)
    target_rows: np.ndarray       # 5 row indices (with scores) for week1..week5
    target_year: int              # used for train/val split


def generate_samples(arrays: RegionDailyArrays,
                     train: pd.DataFrame,
                     cfg: Config) -> Tuple[List[SampleSpec], List[SampleSpec]]:
    """Sliding-window sample generation. Each sample needs:
      - 91 days of daily data ending at end_row
      - 5 future weekly scores at +7, +14, +21, +28, +35 days
    Train/val split: held-out most-recent year per region (if cfg.val_holdout_year).
    Samples whose target year falls before the lookback window are skipped.
    """
    print('Generating sliding-window samples...')
    train_samples: List[SampleSpec] = []
    val_samples:   List[SampleSpec] = []
    win = cfg.window_days

    region_year_max = train.groupby('region_id')['year'].max().to_dict()

    for region, day_idx_arr in arrays.day_idx_arr.items():
        if region not in region_year_max:
            continue
        max_year = region_year_max[region]
        min_year = max_year - cfg.lookback_years + 1
        weekly_rows = arrays.weekly_rows[region]

        for i, end_row in enumerate(weekly_rows):
            if end_row < win - 1:
                continue
            future_idx_pos = i + 1
            if future_idx_pos + 5 > len(weekly_rows):
                continue
            target_rows = weekly_rows[future_idx_pos:future_idx_pos + 5]
            target_year_first = int(day_idx_arr[target_rows[0]] // 372)
            if target_year_first < min_year:
                continue   # outside lookback window

            spec = SampleSpec(region_id=region,
                              end_row=int(end_row),
                              target_rows=target_rows.astype(np.int64),
                              target_year=target_year_first)

            if (cfg.val_holdout_year
                    and target_year_first >= max_year - cfg.val_holdout_years + 1):
                val_samples.append(spec)
            else:
                train_samples.append(spec)

    print(f'  {len(train_samples):,} train samples, {len(val_samples):,} val samples')
    return train_samples, val_samples


# =============================================================================
# Dataset
# =============================================================================

class DroughtDataset(Dataset):
    def __init__(self, samples: List[SampleSpec],
                 arrays: RegionDailyArrays,
                 bundle: FeatureBundle,
                 cfg: Config,
                 subsample_per_region: Optional[int] = None,
                 sample_seed: int = 0):
        self.samples = samples
        self.arrays = arrays
        self.bundle = bundle
        self.cfg = cfg

        if subsample_per_region is not None and subsample_per_region > 0:
            rng = random.Random(sample_seed)
            by_region: Dict[str, List[SampleSpec]] = {}
            for s in samples:
                by_region.setdefault(s.region_id, []).append(s)
            kept: List[SampleSpec] = []
            for r, lst in by_region.items():
                if len(lst) <= subsample_per_region:
                    kept.extend(lst)
                else:
                    kept.extend(rng.sample(lst, subsample_per_region))
            self.samples = kept

        # daily / long-context column count
        self.F = len(bundle.daily_cols)
        self.S = len(bundle.long_ctx_cols)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        region = s.region_id
        end_row = s.end_row
        win = self.cfg.window_days

        # Daily sequence [win, F]
        daily = self.arrays.daily[region][end_row - win + 1 : end_row + 1]   # already standardised

        # Long context: compute from rolling arrays + region static + thresholds
        long_ctx = self._build_long_ctx(region, end_row)

        # Targets
        targets = self.arrays.score_arr[region][s.target_rows]   # shape [5]

        # Target month per week
        target_months = np.array([
            int(self.arrays.day_idx_arr[region][r] // 31 % 12) + 1
            for r in s.target_rows
        ], dtype=np.int64)

        out = {
            'daily': torch.from_numpy(daily),                          # [win, F]
            'long_ctx': torch.from_numpy(long_ctx.astype(np.float32)), # [S]
            'region_idx': torch.tensor(self.bundle.region_to_idx[region], dtype=torch.long),
            'cluster_idx': torch.tensor(int(self.bundle.region_to_cluster.get(region, 0)),
                                         dtype=torch.long),
            'targets': torch.from_numpy(targets.astype(np.float32)),   # [5]
            'target_months': torch.from_numpy(target_months),          # [5]
        }
        return out

    def _build_long_ctx(self, region: str, end_row: int) -> np.ndarray:
        bundle = self.bundle
        cfg = self.cfg
        S = self.S
        if S == 0:
            return np.zeros(0, dtype=np.float32)

        feats = np.zeros(S, dtype=np.float32)
        rolling = self.arrays.rolling[region]
        # month at window end (used for monthly thresholds)
        day_at_end = int(self.arrays.day_idx_arr[region][end_row])
        m_end = (day_at_end % 372) // 31 + 1

        rs = bundle.region_static.get(region, {})
        cluster_id = bundle.region_to_cluster.get(region, 0)
        p91 = float(rolling['precip_91d'][end_row])

        for i, col in enumerate(bundle.long_ctx_cols):
            v = np.nan
            if col in rolling:
                v = float(rolling[col][end_row])
            elif col == 'region_threshold_month':
                v = bundle.region_threshold_by_month.get((region, m_end), np.nan)
            elif col == 'precip_91d_vs_region_threshold':
                v = p91 - rs.get('region_threshold', np.nan)
            elif col == 'precip_91d_vs_region_threshold_month':
                t = bundle.region_threshold_by_month.get((region, m_end), np.nan)
                v = p91 - t
            elif col == 'precip_91d_z_in_region':
                mean = rs.get('region_p91_mean', np.nan)
                std = rs.get('region_p91_std', np.nan)
                if std and not math.isnan(std) and std > 0:
                    v = (p91 - mean) / std
                else:
                    v = 0.0
            elif col == 'cluster_threshold_month':
                v = bundle.cluster_threshold_by_month.get((cluster_id, m_end), np.nan)
            elif col == 'precip_91d_vs_cluster_threshold_month':
                t = bundle.cluster_threshold_by_month.get((cluster_id, m_end), np.nan)
                v = p91 - t
            # --- extra engineered features ---
            elif col == 'vpd_30d':
                v = float(rolling['temp_30d'][end_row]) - float(rolling['dp_tmp_30d'][end_row])
            elif col == 'precip_recent_minus_old':
                p30 = float(rolling['precip_30d'][end_row])
                v = p30 - (p91 - p30) / 2.0
            elif col == 'precip_91d_z_in_region_for_month':
                stats = bundle.region_month_precip_stats.get((region, m_end))
                if stats is not None:
                    mean, std = stats
                    if std > 0 and not math.isnan(std):
                        v = (p91 - mean) / std
                    else:
                        v = 0.0
                else:
                    v = 0.0
            else:
                v = rs.get(col, np.nan)
            feats[i] = v if v == v else 0.0   # NaN -> 0 after standardization
        # standardize
        feats = (feats - bundle.long_ctx_means) / bundle.long_ctx_stds
        return feats


# =============================================================================
# Model
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

    def forward(self, x):  # x: [B, L, D]
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)   # [B, D]


class CrossAttnFusion(nn.Module):
    """One layer of cross-attention: daily-pooled (query) attends to long-context (key/value)."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q, kv: [B, D]  -> treat as length-1 sequences
        q_seq = q.unsqueeze(1)
        kv_seq = kv.unsqueeze(1)
        h, _ = self.attn(q_seq, kv_seq, kv_seq)
        h = self.norm(q_seq + h)
        h = self.norm2(h + self.ffn(h))
        return h.squeeze(1)


class DroughtModel(nn.Module):
    """Drought predictor with three input streams that converge into per-week
    ordinal heads.

    Size-knob cheat sheet (all read from Config):
        d_model           Transformer hidden + daily projection + long_ctx_mlp output dim.
                          The "backbone width". Must be divisible by n_heads.
        n_transformer_layers / n_heads / ffn_dim
                          Transformer-only sizing. Ignored if use_daily_branch=False.
        head_hidden_dim   Width of the post-fusion MLP and the per-week head MLP.
                          Independent of d_model.
        region_emb_dim, cluster_emb_dim, month_emb_dim, week_emb_dim
                          Small categorical embeddings. Concatenated into the fusion
                          (region/cluster) or the per-week head (month/week).
        dropout           Used throughout.
    """
    def __init__(self, cfg: Config, bundle: FeatureBundle):
        super().__init__()
        self.cfg = cfg
        self.bundle = bundle
        F_ = len(bundle.daily_cols)
        S_ = len(bundle.long_ctx_cols)
        D = cfg.d_model                                  # backbone width (Transformer + long_ctx_mlp output)

        # ----- Daily branch (Transformer over the 91-day sequence) -----
        # Skipped entirely when use_daily_branch=False.
        if cfg.use_daily_branch:
            self.daily_proj = nn.Linear(F_, D)            # raw daily feature dim F_ -> d_model
            self.pos_enc = PositionalEncoding(D, max_len=cfg.window_days + 16)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=D,                                # ← d_model
                nhead=cfg.n_heads,                        # ← n_heads
                dim_feedforward=cfg.ffn_dim,              # ← ffn_dim
                dropout=cfg.dropout,
                batch_first=True, norm_first=True,
            )
            self.daily_enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_transformer_layers)
            if cfg.use_attn_pooling:
                self.pool = AttentionPooling(D)           # learned-query pool over 91 tokens -> [B, d_model]
            else:
                self.pool = None
            # Optional: freeze the daily branch at random init. The Transformer
            # then acts as a stochastic feature extractor (dropout still active
            # during training, so its outputs vary per batch) without any of
            # its weights being updated. Used to test whether the Transformer
            # adds anything beyond noise regularization.
            if cfg.freeze_daily_branch:
                for p in self.daily_proj.parameters(): p.requires_grad = False
                for p in self.daily_enc.parameters():  p.requires_grad = False
                if self.pool is not None:
                    for p in self.pool.parameters():   p.requires_grad = False
        else:
            self.daily_proj = None
            self.pos_enc = None
            self.daily_enc = None
            self.pool = None

        # ----- Long-context branch (MLP over ~25 precomputed scalars) -----
        # Output dim is d_model so it can be cross-attended with the daily pool;
        # when daily is disabled the same vector just gets concatenated into fusion.
        if cfg.use_long_context and S_ > 0:
            self.long_ctx_mlp = nn.Sequential(
                nn.Linear(S_, D),                         # 25 scalars -> d_model
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(D, D),                          # d_model -> d_model
            )
        else:
            self.long_ctx_mlp = None

        # identity embeddings
        if cfg.use_region_embedding:
            self.region_emb = nn.Embedding(bundle.n_regions, cfg.region_emb_dim)
        else:
            self.region_emb = None
        if cfg.use_cluster_embedding and bundle.n_clusters > 0:
            self.cluster_emb = nn.Embedding(bundle.n_clusters + 1, cfg.cluster_emb_dim)
        else:
            self.cluster_emb = None

        # ----- Fusion: combine daily / long_ctx / identity into one vector -----
        # Input width depends on which branches are active:
        #     fuse_in = daily_dim + long_ctx_dim + identity_dim
        # where daily/long_ctx contribute D each, and identity is the
        # concatenation of region and cluster embeddings.
        identity_dim = ((cfg.region_emb_dim if self.region_emb is not None else 0)
                         + (cfg.cluster_emb_dim if self.cluster_emb is not None else 0))
        daily_dim    = D if cfg.use_daily_branch else 0
        long_ctx_dim = D if self.long_ctx_mlp is not None else 0
        fuse_in = daily_dim + long_ctx_dim + identity_dim
        # Cross-attention fusion (daily query, long_ctx key/value) only makes
        # sense when both branches exist; otherwise we just concatenate.
        if (cfg.use_cross_attn_fusion and cfg.use_daily_branch
                and self.long_ctx_mlp is not None):
            self.cross_attn = CrossAttnFusion(D, n_heads=cfg.n_heads, dropout=cfg.dropout)
            fuse_in = D + identity_dim   # daily already conditioned by cross-attn
        else:
            self.cross_attn = None
        # Fusion MLP: takes the concatenated input down to head_hidden_dim
        self.fuse_mlp = nn.Sequential(
            nn.Linear(fuse_in, cfg.head_hidden_dim),       # fuse_in -> head_hidden_dim
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim),
        )

        # ----- Per-week prediction head -----
        # The head is SHARED across all 5 future weeks. Per-week differentiation
        # comes only from the target_month and week_index embeddings concatenated
        # into the head's input. Output dim is score_max (5) for ordinal mode
        # (5 logits = P(score>=1), ..., P(score>=5)) or 1 for plain regression.
        #
        # SPECIAL CASE: when cfg.replicate_prediction is True, we treat all 5
        # weeks as a single shared prediction — per-week embeddings are forced
        # OFF and the head runs once per sample. The loss is then computed
        # against the 5 targets simultaneously (one prediction vs 5 truths).
        head_extra = 0
        if cfg.use_target_month_embedding and not cfg.replicate_prediction:
            self.target_month_emb = nn.Embedding(13, cfg.month_emb_dim)        # months 1-12 + 1 unused
            head_extra += cfg.month_emb_dim
        else:
            self.target_month_emb = None
        if cfg.use_week_index_embedding and not cfg.replicate_prediction:
            self.week_emb = nn.Embedding(cfg.n_pred_weeks, cfg.week_emb_dim)   # one vector per future week slot
            head_extra += cfg.week_emb_dim
        else:
            self.week_emb = None
        head_in = cfg.head_hidden_dim + head_extra                              # fuse output + per-week embeddings
        n_out = cfg.score_max if cfg.use_ordinal_head else 1
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, n_out),
        )

    def forward(self,
                daily: torch.Tensor,
                long_ctx: torch.Tensor,
                region_idx: torch.Tensor,
                cluster_idx: torch.Tensor,
                target_months: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, n_pred_weeks, n_out]."""
        B = long_ctx.size(0) if self.daily_proj is None else daily.size(0)

        # daily branch (optional)
        if self.daily_proj is not None:
            h = self.daily_proj(daily)
            h = self.pos_enc(h)
            h = self.daily_enc(h)                 # [B, win, D]
            if self.pool is not None:
                d_pool = self.pool(h)             # [B, D]
            else:
                d_pool = h.mean(dim=1)
        else:
            d_pool = None

        # long-context branch
        if self.long_ctx_mlp is not None:
            lc = self.long_ctx_mlp(long_ctx)      # [B, D]
        else:
            lc = None

        # identity embeddings
        id_parts = []
        if self.region_emb is not None:
            r = self.region_emb(region_idx)
            if self.training and self.cfg.region_emb_dropout > 0:
                mask = (torch.rand(B, 1, device=r.device) > self.cfg.region_emb_dropout).float()
                r = r * mask
            id_parts.append(r)
        if self.cluster_emb is not None:
            id_parts.append(self.cluster_emb(cluster_idx))
        identity = torch.cat(id_parts, dim=1) if id_parts else None

        # fusion
        if self.cross_attn is not None and lc is not None and d_pool is not None:
            d_pool = self.cross_attn(d_pool, lc)
            parts = [d_pool]
        else:
            parts = []
            if d_pool is not None:
                parts.append(d_pool)
            if lc is not None:
                parts.append(lc)
        if identity is not None:
            parts.append(identity)
        if not parts:
            raise RuntimeError('No active branches; at least one of '
                                'use_daily_branch / use_long_context / identity must be enabled')
        x = torch.cat(parts, dim=1)
        x = self.fuse_mlp(x)                      # [B, H]

        # per-week heads (shared MLP, conditioned by week + target month embeddings)
        W = self.cfg.n_pred_weeks
        if self.cfg.replicate_prediction:
            # One head pass, broadcast to all 5 weeks. The 5 outputs are exactly
            # equal — same gradient flows to the single set of weights from the
            # loss against each of the 5 targets.
            out = self.head(x)                              # [B, n_out]
            return out.unsqueeze(1).expand(-1, W, -1).contiguous()
        outs = []
        for k in range(W):
            xs = [x]
            if self.target_month_emb is not None:
                xs.append(self.target_month_emb(target_months[:, k]))
            if self.week_emb is not None:
                week_idx = torch.full((B,), k, dtype=torch.long, device=x.device)
                xs.append(self.week_emb(week_idx))
            outs.append(self.head(torch.cat(xs, dim=1)))   # [B, n_out]
        return torch.stack(outs, dim=1)                     # [B, W, n_out]


# =============================================================================
# Loss
# =============================================================================

def compute_loss(logits: torch.Tensor,
                 targets: torch.Tensor,
                 cfg: Config,
                 horizon_mask: Optional[torch.Tensor] = None,
                 sample_weights: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """logits: [B, W, n_out].  targets: [B, W] float scores 0..5.
    horizon_mask: [W]   -- per-week mask (curriculum).
    sample_weights: [B, W] -- per-sample-per-week loss weight (e.g. for
        importance correction toward test distribution).
    """
    B, W, n_out = logits.shape
    losses: Dict[str, float] = {}

    # combined weight mask: [B, W]
    weight = torch.ones(B, W, device=logits.device)
    if horizon_mask is not None:
        weight = weight * horizon_mask.to(logits.device).float().view(1, W)
    if sample_weights is not None:
        weight = weight * sample_weights

    denom = weight.sum().clamp(min=1e-6)

    if cfg.use_ordinal_head:
        ord_labels = torch.zeros(B, W, n_out, device=logits.device)
        for k in range(n_out):
            ord_labels[..., k] = (targets >= (k + 1)).float()
        bce_raw = F.binary_cross_entropy_with_logits(logits, ord_labels, reduction='none')
        bce_per = bce_raw.sum(dim=-1)                       # [B, W]
        probs = torch.sigmoid(logits)
        expected = probs.sum(dim=-1)                        # [B, W]
        mae_per = (expected - targets).abs()                # [B, W]

        bce = (bce_per * weight).sum() / denom
        mae = (mae_per * weight).sum() / denom
        loss = cfg.bce_weight * bce + cfg.mae_weight * mae
        losses['bce'] = float(bce.item())
        losses['mae_anchor'] = float(mae.item())
    else:
        pred = logits.squeeze(-1)
        mae_per = (pred - targets).abs()
        loss = (mae_per * weight).sum() / denom
        losses['mae'] = float(loss.item())
    return loss, losses


def expected_score(logits: torch.Tensor, cfg: Config) -> torch.Tensor:
    """logits: [..., n_out]. Returns expected score (clipped 0..score_max)."""
    if cfg.use_ordinal_head:
        return torch.sigmoid(logits).sum(dim=-1).clamp(0, cfg.score_max)
    return logits.squeeze(-1).clamp(0, cfg.score_max)


def compute_score_class_weights(score_targets_flat: np.ndarray,
                                 cfg: Config) -> np.ndarray:
    """Given an array of integer score targets, return an array of shape
    [score_max+1] of weights w_k = P_test(k) / P_data(k), clipped to
    [1/sample_weight_clip, sample_weight_clip] and normalized so that
    E_data[w] = 1 (so the loss scale stays the same)."""
    K = cfg.score_max + 1
    targets_int = np.clip(np.round(score_targets_flat).astype(int), 0, cfg.score_max)
    counts = np.bincount(targets_int, minlength=K).astype(np.float64)
    if counts.sum() == 0:
        return np.ones(K, dtype=np.float32)
    data_dist = counts / counts.sum()
    test_dist = np.array(cfg.test_target_distribution, dtype=np.float64)
    if len(test_dist) != K:
        raise ValueError(f'test_target_distribution must have {K} entries, got {len(test_dist)}')
    w = test_dist / np.maximum(data_dist, 1e-9)
    w = np.clip(w, 1.0 / cfg.sample_weight_clip, cfg.sample_weight_clip)
    # normalize so that the expected weight under the data distribution is 1
    expected = (data_dist * w).sum()
    if expected > 0:
        w = w / expected
    return w.astype(np.float32)


# =============================================================================
# Test-feature building (for inference)
# =============================================================================

def build_test_features(test: pd.DataFrame, bundle: FeatureBundle, cfg: Config):
    """Returns dict per region with daily[91, F], long_ctx[S], target_months[5]."""
    out = {}
    daily_cols = bundle.daily_cols
    for region, t in test.groupby('region_id', sort=False):
        t = t.sort_values('day_idx').reset_index(drop=True)
        if len(t) < cfg.window_days:
            continue
        # take last 91 days
        sub = t.iloc[-cfg.window_days:]
        daily = sub[daily_cols].values.astype(np.float32)
        daily = (daily - bundle.daily_means) / bundle.daily_stds

        # long-context
        last = sub.iloc[-1]
        last_day = int(last['day_idx'])
        m_end = int(last['month'])

        # rolling features at window end (from the test window itself)
        precip_91d = float(sub['prec'].sum())
        precip_30d = float(sub['prec'].tail(30).sum())
        precip_7d  = float(sub['prec'].tail( 7).sum())
        temp_30d   = float(sub['tmp'].tail(30).mean())
        temp_7d    = float(sub['tmp'].tail( 7).mean())
        humidity_30d = float(sub['humidity'].tail(30).mean())
        dp_tmp_30d   = float(sub['dp_tmp'].tail(30).mean())
        wind_30d     = float(sub['wind'].tail(30).mean())
        surf_tmp_30d = float(sub['surf_tmp'].tail(30).mean())
        surf_pre_30d = float(sub['surf_pre'].tail(30).mean())

        # extra engineered scalars from the test window itself
        prec_arr = sub['prec'].values
        dry_mask = (prec_arr < 1.0).astype(int)
        cdry = 0
        for x in dry_mask[::-1]:
            if x: cdry += 1
            else: break
        current_dry_streak = float(cdry)
        wet_days_30d = float((sub['prec'].tail(30).values >= 1.0).sum())

        rolling_lookup = {
            'precip_91d': precip_91d, 'precip_30d': precip_30d, 'precip_7d': precip_7d,
            'temp_30d':   temp_30d,   'temp_7d':   temp_7d,
            'humidity_30d': humidity_30d, 'dp_tmp_30d': dp_tmp_30d,
            'wind_30d': wind_30d, 'surf_tmp_30d': surf_tmp_30d, 'surf_pre_30d': surf_pre_30d,
            'current_dry_streak': current_dry_streak,
            'wet_days_30d': wet_days_30d,
        }

        rs = bundle.region_static.get(region, {})
        cluster_id = bundle.region_to_cluster.get(region, 0)

        feats = np.zeros(len(bundle.long_ctx_cols), dtype=np.float32)
        for i, col in enumerate(bundle.long_ctx_cols):
            v = np.nan
            if col in rolling_lookup:
                v = rolling_lookup[col]
            elif col == 'region_threshold_month':
                v = bundle.region_threshold_by_month.get((region, m_end), np.nan)
            elif col == 'precip_91d_vs_region_threshold':
                v = precip_91d - rs.get('region_threshold', np.nan)
            elif col == 'precip_91d_vs_region_threshold_month':
                t_v = bundle.region_threshold_by_month.get((region, m_end), np.nan)
                v = precip_91d - t_v
            elif col == 'precip_91d_z_in_region':
                mean = rs.get('region_p91_mean', np.nan)
                std = rs.get('region_p91_std', np.nan)
                if std and not math.isnan(std) and std > 0:
                    v = (precip_91d - mean) / std
                else:
                    v = 0.0
            elif col == 'cluster_threshold_month':
                v = bundle.cluster_threshold_by_month.get((cluster_id, m_end), np.nan)
            elif col == 'precip_91d_vs_cluster_threshold_month':
                t_v = bundle.cluster_threshold_by_month.get((cluster_id, m_end), np.nan)
                v = precip_91d - t_v
            # --- extra engineered features ---
            elif col == 'vpd_30d':
                v = temp_30d - dp_tmp_30d
            elif col == 'precip_recent_minus_old':
                v = precip_30d - (precip_91d - precip_30d) / 2.0
            elif col == 'precip_91d_z_in_region_for_month':
                stats = bundle.region_month_precip_stats.get((region, m_end))
                if stats is not None:
                    mean, std = stats
                    if std > 0 and not math.isnan(std):
                        v = (precip_91d - mean) / std
                    else:
                        v = 0.0
                else:
                    v = 0.0
            else:
                v = rs.get(col, np.nan)
            feats[i] = v if v == v else 0.0
        long_ctx = (feats - bundle.long_ctx_means) / bundle.long_ctx_stds

        # target months: 5 weeks after window end
        target_months = []
        for k in range(1, cfg.n_pred_weeks + 1):
            d_next = last_day + 7 * k
            target_months.append(((d_next % 372) // 31) + 1)
        target_months = np.array(target_months, dtype=np.int64)

        out[region] = {
            'daily': torch.from_numpy(daily),
            'long_ctx': torch.from_numpy(long_ctx.astype(np.float32)),
            'region_idx': torch.tensor(bundle.region_to_idx[region], dtype=torch.long),
            'cluster_idx': torch.tensor(int(cluster_id), dtype=torch.long),
            'target_months': torch.from_numpy(target_months),
        }
    return out


# =============================================================================
# Training utilities
# =============================================================================

def horizon_mask_for_epoch(cfg: Config, epoch: int) -> torch.Tensor:
    """Curriculum: enable progressively more horizons. Always returns float [W]."""
    W = cfg.n_pred_weeks
    if not cfg.use_curriculum:
        return torch.ones(W)
    weeks_active = min(W, 1 + epoch // max(1, cfg.curriculum_epochs))
    mask = torch.zeros(W)
    mask[:weeks_active] = 1
    return mask


def mixup_batch(batch: Dict[str, torch.Tensor], cfg: Config) -> Dict[str, torch.Tensor]:
    """Within-cluster mixup. Mixes daily, long_ctx, targets pairwise."""
    if cfg.mixup_alpha <= 0:
        return batch
    B = batch['daily'].size(0)
    # find pairing (within-cluster preferred)
    perm = torch.randperm(B)
    if cfg.mixup_within_cluster_only:
        clusters = batch['cluster_idx'].cpu().numpy()
        cluster_to_indices: Dict[int, List[int]] = {}
        for i, c in enumerate(clusters):
            cluster_to_indices.setdefault(int(c), []).append(i)
        new_perm = list(range(B))
        for i in range(B):
            c = int(clusters[i])
            pool = cluster_to_indices.get(c, [])
            if len(pool) >= 2:
                new_perm[i] = pool[random.randrange(len(pool))]
        perm = torch.tensor(new_perm, dtype=torch.long)
    lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
    out = {
        'daily':         lam * batch['daily'] + (1 - lam) * batch['daily'][perm],
        'long_ctx':      lam * batch['long_ctx'] + (1 - lam) * batch['long_ctx'][perm],
        'region_idx':    batch['region_idx'],     # keep primary identity
        'cluster_idx':   batch['cluster_idx'],
        'target_months': batch['target_months'],   # keep primary
        'targets':       lam * batch['targets'] + (1 - lam) * batch['targets'][perm],
    }
    return out


def add_label_noise(targets: torch.Tensor, cfg: Config) -> torch.Tensor:
    if not cfg.use_label_noise or cfg.label_noise_std <= 0:
        return targets
    noise = torch.randn_like(targets) * cfg.label_noise_std
    return (targets + noise).clamp(0, cfg.score_max)


# =============================================================================
# Train / eval loops
# =============================================================================

def evaluate(model: DroughtModel, loader: DataLoader, cfg: Config, device: str,
             log_distribution: bool = True) -> Dict[str, float]:
    """Returns val metrics + optionally logs prediction-distribution stats
    (mean / std overall and per week) so you can spot mean-collapse."""
    model.eval()
    total_mae = 0.0
    total_n = 0
    week_mae = np.zeros(cfg.n_pred_weeks)
    week_n = np.zeros(cfg.n_pred_weeks)
    pred_chunks: List[np.ndarray] = []
    tgt_chunks:  List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            daily = batch['daily'].to(device, non_blocking=True)
            long_ctx = batch['long_ctx'].to(device, non_blocking=True)
            region_idx = batch['region_idx'].to(device)
            cluster_idx = batch['cluster_idx'].to(device)
            target_months = batch['target_months'].to(device)
            targets = batch['targets'].to(device)

            logits = model(daily, long_ctx, region_idx, cluster_idx, target_months)
            pred = expected_score(logits, cfg)            # [B, W]
            err = (pred - targets).abs()
            total_mae += err.sum().item()
            total_n += err.numel()
            for k in range(cfg.n_pred_weeks):
                week_mae[k] += err[:, k].sum().item()
                week_n[k] += err.size(0)
            if log_distribution:
                pred_chunks.append(pred.cpu().numpy())
                tgt_chunks.append(targets.cpu().numpy())

    out = {'val_mae': total_mae / max(total_n, 1)}
    for k in range(cfg.n_pred_weeks):
        out[f'val_mae_week{k+1}'] = float(week_mae[k] / max(week_n[k], 1))

    # NOTE: we no longer print here. Diagnostic lines are stashed in
    # out['_diag_lines'] and printed by the caller AFTER the epoch header,
    # so each epoch's block reads top-to-bottom in order.
    diag_lines: List[str] = []
    if log_distribution and pred_chunks:
        preds   = np.concatenate(pred_chunks, axis=0)     # [N, W]
        targets = np.concatenate(tgt_chunks,  axis=0)     # [N, W]
        p_mean, p_std = float(preds.mean()),   float(preds.std())
        t_mean, t_std = float(targets.mean()), float(targets.std())
        baseline_mae = float(np.abs(targets - t_mean).mean())
        q = np.quantile(preds, [0.05, 0.5, 0.95])
        diag_lines.append(f'  pred dist  mean={p_mean:.3f}  std={p_std:.3f}  '
                          f'q05={q[0]:.2f}  q50={q[1]:.2f}  q95={q[2]:.2f}')
        diag_lines.append(f'  val target dist mean={t_mean:.3f}  std={t_std:.3f}  '
                          f'(always-predict-mean baseline MAE = {baseline_mae:.3f})')
        per_w = '  '.join(f'w{k+1}:m={preds[:, k].mean():.3f}/s={preds[:, k].std():.3f}'
                          for k in range(cfg.n_pred_weeks))
        diag_lines.append(f'  per-week pred {per_w}')
        out['_pred_mean'] = p_mean
        out['_pred_std']  = p_std
        out['_baseline_mae'] = baseline_mae

        # Importance-weighted val MAE (estimate of test-distribution MAE).
        if cfg.use_test_dist_weighted_val:
            K = cfg.score_max + 1
            target_int = np.clip(np.round(targets).astype(int), 0, K - 1)
            val_counts = np.bincount(target_int.flatten(), minlength=K).astype(np.float64)
            val_dist = val_counts / max(val_counts.sum(), 1.0)
            test_dist = np.array(cfg.test_target_distribution, dtype=np.float64)
            w_per_score = test_dist / np.maximum(val_dist, 1e-9)
            w_per_score = np.clip(w_per_score, 1.0 / cfg.sample_weight_clip, cfg.sample_weight_clip)
            per_sample_w = w_per_score[target_int]              # [N, W]
            errors = np.abs(preds - targets)
            denom = per_sample_w.sum()
            weighted_mae = float((errors * per_sample_w).sum() / max(denom, 1e-9))
            week_w_mae = []
            for k in range(cfg.n_pred_weeks):
                d = per_sample_w[:, k].sum()
                week_w_mae.append(float((errors[:, k] * per_sample_w[:, k]).sum() / max(d, 1e-9)))
            out['val_mae_weighted'] = weighted_mae
            for k, m in enumerate(week_w_mae):
                out[f'val_mae_weighted_week{k+1}'] = m
            diag_lines.append(
                f'  per-week val_mae_weighted: ' +
                ' '.join(f'w{k+1}={m:.3f}' for k, m in enumerate(week_w_mae)))
            out['_score_weights_val'] = w_per_score.tolist()
    out['_diag_lines'] = diag_lines
    return out


def train(cfg: Config):
    device = cfg.device if torch.cuda.is_available() or cfg.device == 'cpu' else 'cpu'
    if device != cfg.device:
        print(f'WARNING: requested device {cfg.device} unavailable, using {device}')
    print(f'=== train run "{cfg.run_name}" on {device} ===')
    set_seed(cfg.seed)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # 1. data
    train_df, test_df, sample_df = load_raw_data(cfg)
    region_to_cluster, n_clusters = load_clusters(cfg)
    bundle = build_feature_bundle(cfg, train_df, test_df, region_to_cluster, n_clusters)
    arrays = build_region_arrays(train_df, bundle, cfg)

    # filter to lookback years for sample generation
    rmax = train_df.groupby('region_id')['year'].max().rename('max_year')
    tr_view = train_df.merge(rmax, left_on='region_id', right_index=True)
    tr_view = tr_view[tr_view['year'] >= tr_view['max_year'] - cfg.lookback_years + 1]
    train_samples, val_samples = generate_samples(arrays, tr_view, cfg)

    # 2. datasets / loaders
    train_ds = DroughtDataset(train_samples, arrays, bundle, cfg,
                              subsample_per_region=cfg.samples_per_region_per_epoch,
                              sample_seed=cfg.seed)
    val_ds = DroughtDataset(val_samples, arrays, bundle, cfg)
    print(f'  train epoch size: {len(train_ds):,}  val: {len(val_ds):,}')

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    # 3. model
    model = DroughtModel(cfg, bundle).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable != n_params:
        print(f'Model: {n_params:,} parameters ({n_trainable:,} trainable, '
              f'{n_params - n_trainable:,} frozen)')
    else:
        print(f'Model: {n_params:,} parameters')

    # 4. region embedding init from cluster centroid (averaging existing region init)
    if (cfg.use_region_embedding and cfg.use_cluster_embedding
            and cfg.region_emb_init_from_cluster and model.cluster_emb is not None):
        with torch.no_grad():
            # average cluster_emb with region_emb (project cluster_emb to region_emb_dim)
            # simple init: copy cluster_emb (truncate or zero-pad) into each region's embedding
            r_dim = cfg.region_emb_dim
            c_dim = cfg.cluster_emb_dim
            base_dim = min(r_dim, c_dim)
            for region, ridx in bundle.region_to_idx.items():
                cid = bundle.region_to_cluster.get(region, 0)
                c_vec = model.cluster_emb.weight[cid][:base_dim]
                model.region_emb.weight[ridx][:base_dim] = c_vec.clone()

    # 5. optim
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.n_epochs)
    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # 5b. score-class weight tensor for distribution-matched training
    score_weight_tensor = None
    if cfg.use_test_dist_sample_weights:
        all_train_targets = np.concatenate([
            arrays.score_arr[s.region_id][s.target_rows] for s in train_samples
        ])
        all_train_targets = all_train_targets[~np.isnan(all_train_targets)]
        w_arr = compute_score_class_weights(all_train_targets, cfg)
        score_weight_tensor = torch.tensor(w_arr, dtype=torch.float32, device=device)
        print(f'Per-score training weights (test/train ratio): '
              f'{[round(float(x), 3) for x in w_arr.tolist()]}')

    # 6. training loop
    best_val = float('inf')
    global_step = 0
    daily_branch_frozen_runtime = False   # tracks freeze_daily_branch_after_epoch
    for epoch in range(1, cfg.n_epochs + 1):
        # Freeze the daily branch starting at the configured epoch (if any).
        # Triggered once at the start of the matching epoch.
        if (cfg.freeze_daily_branch_after_epoch > 0
                and not daily_branch_frozen_runtime
                and epoch > cfg.freeze_daily_branch_after_epoch
                and model.daily_proj is not None):
            for p in model.daily_proj.parameters(): p.requires_grad = False
            for p in model.daily_enc.parameters():  p.requires_grad = False
            if model.pool is not None:
                for p in model.pool.parameters():   p.requires_grad = False
            daily_branch_frozen_runtime = True
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'  *** Froze daily branch at start of epoch {epoch} '
                  f'({n_trainable:,} parameters still trainable) ***')
        hm = horizon_mask_for_epoch(cfg, epoch - 1)
        model.train()
        ep_loss = 0.0
        ep_steps = 0
        t0 = time.time()
        # rebuild train_ds with new subsample each epoch (variety)
        train_ds = DroughtDataset(train_samples, arrays, bundle, cfg,
                                   subsample_per_region=cfg.samples_per_region_per_epoch,
                                   sample_seed=cfg.seed + epoch)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                   num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
        for batch in train_loader:
            if cfg.use_mixup and cfg.mixup_alpha > 0:
                batch = mixup_batch(batch, cfg)
            batch['targets'] = add_label_noise(batch['targets'], cfg)
            daily = batch['daily'].to(device, non_blocking=True)
            long_ctx = batch['long_ctx'].to(device, non_blocking=True)
            region_idx = batch['region_idx'].to(device)
            cluster_idx = batch['cluster_idx'].to(device)
            target_months = batch['target_months'].to(device)
            targets = batch['targets'].to(device)
            sample_weights = None
            if score_weight_tensor is not None:
                idx = targets.round().long().clamp(0, cfg.score_max)
                sample_weights = score_weight_tensor[idx]   # [B, W]
            logits = model(daily, long_ctx, region_idx, cluster_idx, target_months)
            loss, _ = compute_loss(logits, targets, cfg, horizon_mask=hm,
                                    sample_weights=sample_weights)
            optim.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            scheduler.step()
            ep_loss += loss.item()
            ep_steps += 1
            global_step += 1
            if cfg.print_every_n_steps > 0 and global_step % cfg.print_every_n_steps == 0:
                print(f'  ep{epoch:02d} step{global_step:06d} loss={loss.item():.4f}')

        avg = ep_loss / max(ep_steps, 1)
        # val
        metrics = evaluate(model, val_loader, cfg, device)
        val_mae = metrics['val_mae']
        val_mae_w = metrics.get('val_mae_weighted')

        # 'best' criterion: weighted MAE if available (closer to test), else plain val_mae
        best_metric = val_mae_w if (cfg.use_test_dist_weighted_val and val_mae_w is not None) else val_mae
        best_metric_name = 'val_mae_w' if (cfg.use_test_dist_weighted_val and val_mae_w is not None) else 'val_mae'
        is_new_best = cfg.save_best and best_metric < best_val

        # ===== prominent epoch header (val_mae_w highlighted) =====
        weeks_str = ' '.join(f'w{k+1}={metrics[f"val_mae_week{k+1}"]:.3f}'
                              for k in range(cfg.n_pred_weeks))
        best_tag = '   ★★★ NEW BEST ★★★' if is_new_best else ''
        if val_mae_w is not None:
            headline = f'>>>  val_mae_w = {val_mae_w:.4f}  <<<'
        else:
            headline = f'>>>  val_mae = {val_mae:.4f}  <<<'
        print()   # blank line so it pops in scrolling output
        print(f'╔══ epoch {epoch:02d}  {headline}{best_tag}')
        print(f'║   val_mae={val_mae:.4f}  train_loss={avg:.4f}  '
              f'per-week: {weeks_str}  ({time.time() - t0:.1f}s)')
        # Diagnostic lines now print AFTER the header so each epoch block reads top-down
        for line in metrics.get('_diag_lines', []):
            print(line)

        # checkpoint
        ckpt = {
            'model_state': model.state_dict(),
            'config':      asdict(cfg),
            'bundle':      _serialize_bundle(bundle),
            'epoch':       epoch,
            'val_mae':     val_mae,
            'val_mae_weighted': val_mae_w,
            'best_metric': float(best_metric),
            'best_metric_name': best_metric_name,
        }
        if is_new_best:
            best_val = best_metric
            torch.save(ckpt, Path(cfg.checkpoint_dir) / f'{cfg.run_name}_best.pt')
        if epoch % cfg.save_every_n_epochs == 0 or epoch == cfg.n_epochs:
            torch.save(ckpt, Path(cfg.checkpoint_dir) / f'{cfg.run_name}_ep{epoch}.pt')

    # 7. final submission with best (or last) checkpoint
    loaded_epoch = None
    loaded_val_mae = None
    loaded_val_mae_w = None
    loaded_source = None
    best_ckpt_path = Path(cfg.checkpoint_dir) / f'{cfg.run_name}_best.pt'
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        loaded_epoch    = ckpt.get('epoch')
        loaded_val_mae  = ckpt.get('val_mae')
        loaded_val_mae_w = ckpt.get('val_mae_weighted')
        loaded_source   = str(best_ckpt_path)
        print(f'Reloaded best model for submission (epoch={loaded_epoch}, '
              f'val_mae={loaded_val_mae:.4f})')

    write_submission(model, bundle, test_df, sample_df, cfg, device,
                      epoch=loaded_epoch, source_checkpoint=loaded_source,
                      val_mae=loaded_val_mae, val_mae_weighted=loaded_val_mae_w)


# =============================================================================
# Checkpoint serialization
# =============================================================================

def _serialize_bundle(bundle: FeatureBundle) -> Dict:
    return {
        'daily_cols': bundle.daily_cols,
        'long_ctx_cols': bundle.long_ctx_cols,
        'daily_means': bundle.daily_means.tolist(),
        'daily_stds': bundle.daily_stds.tolist(),
        'long_ctx_means': bundle.long_ctx_means.tolist(),
        'long_ctx_stds': bundle.long_ctx_stds.tolist(),
        'region_to_idx': bundle.region_to_idx,
        'region_to_cluster': bundle.region_to_cluster,
        'n_regions': bundle.n_regions,
        'n_clusters': bundle.n_clusters,
        'region_static': bundle.region_static,
        'region_threshold_by_month': {f'{r}|{m}': v
                                       for (r, m), v in bundle.region_threshold_by_month.items()},
        'cluster_threshold_by_month': {f'{c}|{m}': v
                                         for (c, m), v in bundle.cluster_threshold_by_month.items()},
        'region_month_precip_stats': {f'{r}|{m}': list(v)
                                       for (r, m), v in bundle.region_month_precip_stats.items()},
    }


def _deserialize_bundle(blob: Dict) -> FeatureBundle:
    rtm = {}
    for k, v in blob['region_threshold_by_month'].items():
        r, m = k.split('|')
        rtm[(r, int(m))] = v
    ctm = {}
    for k, v in blob['cluster_threshold_by_month'].items():
        c, m = k.split('|')
        ctm[(int(c), int(m))] = v
    rmps = {}
    for k, v in blob.get('region_month_precip_stats', {}).items():
        r, m = k.split('|')
        rmps[(r, int(m))] = (float(v[0]), float(v[1]))
    return FeatureBundle(
        daily_cols=blob['daily_cols'],
        long_ctx_cols=blob['long_ctx_cols'],
        daily_means=np.array(blob['daily_means'], dtype=np.float32),
        daily_stds=np.array(blob['daily_stds'], dtype=np.float32),
        long_ctx_means=np.array(blob['long_ctx_means'], dtype=np.float32),
        long_ctx_stds=np.array(blob['long_ctx_stds'], dtype=np.float32),
        region_to_idx=blob['region_to_idx'],
        region_to_cluster=blob['region_to_cluster'],
        n_regions=blob['n_regions'],
        n_clusters=blob['n_clusters'],
        region_static=blob['region_static'],
        region_threshold_by_month=rtm,
        cluster_threshold_by_month=ctm,
        region_month_precip_stats=rmps,
    )


# =============================================================================
# Submission writing (with optional TTA)
# =============================================================================

@torch.no_grad()
def predict_test(model: DroughtModel,
                 bundle: FeatureBundle,
                 test_df: pd.DataFrame,
                 cfg: Config,
                 device: str) -> Dict[str, np.ndarray]:
    model.eval()
    feats = build_test_features(test_df, bundle, cfg)
    regions = list(feats.keys())
    preds: Dict[str, np.ndarray] = {}

    # base pass
    base_pred = _forward_test(model, feats, regions, cfg, device)
    accum = base_pred
    n_passes = 1

    # TTA: shift the test window by up to ±2 days (uses test data only; no train leakage)
    if cfg.use_tta and cfg.n_tta_augments > 0:
        for k in range(cfg.n_tta_augments):
            shift = (k % 2 + 1) * (1 if k < 2 else -1)
            jit_feats = _build_test_features_shifted(test_df, bundle, cfg, shift)
            jit_pred = _forward_test(model, jit_feats, regions, cfg, device)
            accum = accum + jit_pred
            n_passes += 1
    avg = accum / n_passes
    if cfg.prediction_scale != 1.0:
        avg = np.clip(avg * cfg.prediction_scale, 0, cfg.score_max)
        print(f'  applied prediction_scale={cfg.prediction_scale}')
    for i, r in enumerate(regions):
        preds[r] = avg[i]
    return preds


def _forward_test(model, feats, regions, cfg, device) -> np.ndarray:
    batch_size = cfg.batch_size
    all_pred = []
    for start in range(0, len(regions), batch_size):
        chunk = regions[start:start + batch_size]
        daily = torch.stack([feats[r]['daily']        for r in chunk]).to(device)
        long_ctx = torch.stack([feats[r]['long_ctx']  for r in chunk]).to(device)
        region_idx = torch.stack([feats[r]['region_idx'] for r in chunk]).to(device)
        cluster_idx = torch.stack([feats[r]['cluster_idx'] for r in chunk]).to(device)
        target_months = torch.stack([feats[r]['target_months'] for r in chunk]).to(device)
        logits = model(daily, long_ctx, region_idx, cluster_idx, target_months)
        pred = expected_score(logits, cfg).cpu().numpy()
        all_pred.append(pred)
    return np.concatenate(all_pred, axis=0)


def _build_test_features_shifted(test_df: pd.DataFrame, bundle: FeatureBundle,
                                  cfg: Config, shift_days: int) -> Dict:
    """Drop the last `shift_days` days (or first, for negative shift) to jitter the window."""
    out = {}
    daily_cols = bundle.daily_cols
    for region, t in test_df.groupby('region_id', sort=False):
        t = t.sort_values('day_idx').reset_index(drop=True)
        if shift_days > 0:
            sub = t.iloc[-(cfg.window_days + shift_days):-shift_days]
        else:
            sub = t.iloc[-cfg.window_days + shift_days:] if shift_days < 0 else t.iloc[-cfg.window_days:]
        if len(sub) < cfg.window_days:
            sub = t.iloc[-cfg.window_days:]
        daily = sub[daily_cols].values.astype(np.float32)
        daily = (daily - bundle.daily_means) / bundle.daily_stds

        last_day = int(sub.iloc[-1]['day_idx'])
        m_end = (last_day % 372) // 31 + 1

        precip_91d = float(sub['prec'].sum())
        precip_30d = float(sub['prec'].tail(30).sum())
        precip_7d  = float(sub['prec'].tail(7).sum())
        temp_30d   = float(sub['tmp'].tail(30).mean())
        dp_tmp_30d = float(sub['dp_tmp'].tail(30).mean())
        # extra engineered scalars from the (possibly shifted) test window
        prec_arr = sub['prec'].values
        dry_mask = (prec_arr < 1.0).astype(int)
        cdry = 0
        for x in dry_mask[::-1]:
            if x: cdry += 1
            else: break
        current_dry_streak = float(cdry)
        wet_days_30d = float((sub['prec'].tail(30).values >= 1.0).sum())

        rs = bundle.region_static.get(region, {})
        cluster_id = bundle.region_to_cluster.get(region, 0)
        feats = np.zeros(len(bundle.long_ctx_cols), dtype=np.float32)
        rolling_lookup = {
            'precip_91d': precip_91d,
            'precip_30d': precip_30d,
            'precip_7d':  precip_7d,
            'temp_30d':   temp_30d,
            'temp_7d':    float(sub['tmp'].tail(7).mean()),
            'humidity_30d': float(sub['humidity'].tail(30).mean()),
            'dp_tmp_30d':   dp_tmp_30d,
            'wind_30d':     float(sub['wind'].tail(30).mean()),
            'surf_tmp_30d': float(sub['surf_tmp'].tail(30).mean()),
            'surf_pre_30d': float(sub['surf_pre'].tail(30).mean()),
            'current_dry_streak': current_dry_streak,
            'wet_days_30d': wet_days_30d,
        }
        for i, col in enumerate(bundle.long_ctx_cols):
            if col in rolling_lookup:
                v = rolling_lookup[col]
            elif col == 'region_threshold_month':
                v = bundle.region_threshold_by_month.get((region, m_end), np.nan)
            elif col == 'precip_91d_vs_region_threshold':
                v = precip_91d - rs.get('region_threshold', np.nan)
            elif col == 'precip_91d_vs_region_threshold_month':
                t_v = bundle.region_threshold_by_month.get((region, m_end), np.nan)
                v = precip_91d - t_v
            elif col == 'precip_91d_z_in_region':
                mean = rs.get('region_p91_mean', np.nan)
                std = rs.get('region_p91_std', np.nan)
                v = (precip_91d - mean) / std if (std and not math.isnan(std) and std > 0) else 0.0
            elif col == 'cluster_threshold_month':
                v = bundle.cluster_threshold_by_month.get((cluster_id, m_end), np.nan)
            elif col == 'precip_91d_vs_cluster_threshold_month':
                t_v = bundle.cluster_threshold_by_month.get((cluster_id, m_end), np.nan)
                v = precip_91d - t_v
            elif col == 'vpd_30d':
                v = temp_30d - dp_tmp_30d
            elif col == 'precip_recent_minus_old':
                v = precip_30d - (precip_91d - precip_30d) / 2.0
            elif col == 'precip_91d_z_in_region_for_month':
                stats = bundle.region_month_precip_stats.get((region, m_end))
                if stats is not None:
                    mean, std = stats
                    v = (precip_91d - mean) / std if (std > 0 and not math.isnan(std)) else 0.0
                else:
                    v = 0.0
            else:
                v = rs.get(col, np.nan)
            feats[i] = v if v == v else 0.0
        long_ctx = (feats - bundle.long_ctx_means) / bundle.long_ctx_stds

        target_months = []
        for k in range(1, cfg.n_pred_weeks + 1):
            d_next = last_day + 7 * k
            target_months.append(((d_next % 372) // 31) + 1)
        target_months = np.array(target_months, dtype=np.int64)

        out[region] = {
            'daily': torch.from_numpy(daily),
            'long_ctx': torch.from_numpy(long_ctx.astype(np.float32)),
            'region_idx': torch.tensor(bundle.region_to_idx[region], dtype=torch.long),
            'cluster_idx': torch.tensor(int(cluster_id), dtype=torch.long),
            'target_months': torch.from_numpy(target_months),
        }
    return out


def _next_numbered_path(base_path: str) -> Tuple[Path, int]:
    """Insert _NNN before the extension; find the lowest unused number so
    repeated runs don't overwrite each other. Returns (path, n)."""
    p = Path(base_path)
    stem, suffix, parent = p.stem, p.suffix, p.parent
    parent.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidate = parent / f'{stem}_{n:03d}{suffix}'
        if not candidate.exists():
            return candidate, n
        n += 1


def _write_params_file(params_path: Path,
                        cfg: Config,
                        submission_file: Path,
                        epoch: Optional[int] = None,
                        source_checkpoint: Optional[str] = None,
                        val_mae: Optional[float] = None,
                        val_mae_weighted: Optional[float] = None):
    """Companion file describing exactly how a submission CSV was produced.

    Top of file has the human-essentials (run_name, epoch, source ckpt, val
    metrics). The full Config follows, sorted alphabetically, so the params
    file can fully reconstruct the run later if needed.
    """
    lines: List[str] = []
    lines.append(f'# Submission params')
    lines.append(f'# Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('')
    lines.append(f'submission_file: {submission_file.name}')
    lines.append(f'run_name: {cfg.run_name}')
    if epoch is not None:
        lines.append(f'checkpoint_epoch: {epoch}')
    if source_checkpoint is not None:
        lines.append(f'source_checkpoint: {source_checkpoint}')
    if val_mae is not None:
        lines.append(f'checkpoint_val_mae: {val_mae:.6f}')
    if val_mae_weighted is not None:
        lines.append(f'checkpoint_val_mae_weighted: {val_mae_weighted:.6f}')
    lines.append('')
    lines.append('# === Full Config ===')
    cfg_dict = asdict(cfg)
    for k in sorted(cfg_dict.keys()):
        lines.append(f'{k}: {cfg_dict[k]}')
    params_path.write_text('\n'.join(lines) + '\n')


def write_submission(model: DroughtModel,
                     bundle: FeatureBundle,
                     test_df: pd.DataFrame,
                     sample_df: pd.DataFrame,
                     cfg: Config,
                     device: str,
                     epoch: Optional[int] = None,
                     source_checkpoint: Optional[str] = None,
                     val_mae: Optional[float] = None,
                     val_mae_weighted: Optional[float] = None):
    preds = predict_test(model, bundle, test_df, cfg, device)
    rows = []
    for r in sample_df['region_id']:
        if r in preds:
            p = preds[r]
            if cfg.replicate_prediction:
                v = float(p.mean())
                rows.append([r, v, v, v, v, v])
            else:
                rows.append([r] + [float(x) for x in p])
        else:
            rows.append([r] + [0.0] * cfg.n_pred_weeks)
    cols = ['region_id'] + [f'pred_week{k+1}' for k in range(cfg.n_pred_weeks)]
    sub = pd.DataFrame(rows, columns=cols)
    out_path, n = _next_numbered_path(cfg.submission_path)
    sub.to_csv(out_path, index=False)
    print(f'Wrote {out_path}  ({len(sub)} regions)')

    # Companion params file with the same serial number
    params_path = out_path.parent / f'params_{n:03d}.txt'
    _write_params_file(params_path, cfg, submission_file=out_path,
                        epoch=epoch, source_checkpoint=source_checkpoint,
                        val_mae=val_mae, val_mae_weighted=val_mae_weighted)
    print(f'Wrote {params_path}')


# =============================================================================
# Predict from checkpoint
# =============================================================================

def predict_from_checkpoint(ckpt_path: str, output_path: Optional[str] = None,
                             override_replicate: Optional[bool] = None,
                             override_scale: Optional[float] = None):
    print(f'Loading checkpoint {ckpt_path}...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ckpt['config'])
    if output_path is not None:
        cfg.submission_path = output_path
    else:
        # Always write to submissions/ by default — even if the saved checkpoint's
        # config has an older bare-filename default.
        p = Path(cfg.submission_path)
        if p.parent in (Path('.'), Path('')):
            cfg.submission_path = str(Path('submissions') / p.name)
    if override_replicate is not None:
        cfg.replicate_prediction = override_replicate
    if override_scale is not None:
        cfg.prediction_scale = override_scale
    bundle = _deserialize_bundle(ckpt['bundle'])

    val_mae = ckpt.get('val_mae')
    val_mae_w = ckpt.get('val_mae_weighted')
    info_parts = [f'run "{cfg.run_name}"']
    if ckpt.get('epoch') is not None:
        info_parts.append(f'epoch={ckpt.get("epoch")}')
    if val_mae is not None:
        info_parts.append(f'val_mae={val_mae:.4f}')
    if val_mae_w is not None:
        info_parts.append(f'val_mae_w={val_mae_w:.4f}')
    print(f'Loaded checkpoint  ({", ".join(info_parts)})')

    # Peek at what the next numbered output path will be so the user knows up front.
    # (Same call inside write_submission later returns the same path — _next_numbered_path
    # is idempotent as long as no one else writes between this call and the actual write.)
    peeked_out, _ = _next_numbered_path(cfg.submission_path)
    print(f'Will write submission to: {peeked_out}')

    model = DroughtModel(cfg, bundle).to(device)
    model.load_state_dict(ckpt['model_state'])

    _, test_df, sample_df = load_raw_data(cfg)
    write_submission(model, bundle, test_df, sample_df, cfg, device,
                      epoch=ckpt.get('epoch'),
                      source_checkpoint=str(ckpt_path),
                      val_mae=val_mae,
                      val_mae_weighted=val_mae_w)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Deep drought predictor')
    sub = parser.add_subparsers(dest='cmd', required=True)

    pt = sub.add_parser('train', help='train + write submission')
    pt.add_argument('--cfg', action='append', default=[],
                    help='Override config: --cfg key=value (repeatable)')

    pp = sub.add_parser('predict', help='load checkpoint and write submission')
    pp.add_argument('--checkpoint', required=True)
    pp.add_argument('--output', default=None)
    pp.add_argument('--replicate', type=lambda s: s.lower() in ('1', 'true', 'yes'),
                    default=None, help='override replicate_prediction')
    pp.add_argument('--scale', type=float, default=None,
                    help='multiply final predictions by this scalar (e.g. 0.9)')

    args = parser.parse_args()
    if args.cmd == 'train':
        cfg = cfg_from_cli_overrides(Config(), args.cfg)
        # echo config
        print('Config:')
        for k, v in asdict(cfg).items():
            print(f'  {k}: {v}')
        train(cfg)
    elif args.cmd == 'predict':
        predict_from_checkpoint(args.checkpoint, args.output, args.replicate, args.scale)


if __name__ == '__main__':
    main()
