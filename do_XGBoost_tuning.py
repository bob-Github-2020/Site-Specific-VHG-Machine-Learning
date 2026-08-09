#!/usr/bin/env python3
"""
XGBOOST HYPERPARAMETER TUNING DIAGNOSTIC
==========================================
Tests key parameter combinations to find if we can improve
over the current default configuration (CV R² ≈ 0.49).

Tests combinations of:
  max_depth:      3, 4, 5, 6
  n_estimators:   500, 1000
  learning_rate:  0.05, 0.02
  min_child_weight: 3, 5, 10

Uses the same data pipeline as the production program:
  - All 6 periods pooled, uniform weighting
  - 14 features (9 original + 5 neighborhood)
  - Upstream-cleaned VHG data
  - 5-fold CV for honest evaluation
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.spatial import cKDTree
import warnings
import itertools
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION (same as production program)
# ============================================================================
TARGET_PERIOD = "2020-2025"

ALL_PERIODS = [
    "1970-1975", "1980-1985", "1990-1995",
    "2000-2005", "2010-2015", "2020-2025",
]

HEAD_FILE_PATTERN = "Mean_GWL_{period}.txt"
VHG_FILE_PATTERN = "VHG_pairwise_{period}.csv"

BOUNDS = [-96.0, -94.8, 29.0, 30.4]
FT_TO_M = 0.3048
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON_HOUSTON = 111.0 * np.cos(np.radians(29.7))

N_NEAREST_WELLS = 5
LOCAL_RADIUS_KM = 10.0

VHG_MIN = -0.15
VHG_MAX = 0.15

# ============================================================================
# PARAMETER GRID TO TEST
# ============================================================================
PARAM_GRID = {
    'max_depth':        [3, 4, 5, 6],
    'n_estimators':     [500, 1000],
    'learning_rate':    [0.05, 0.02],
    'min_child_weight': [3, 5, 10],
}

# Fixed parameters (not tuned)
FIXED_PARAMS = {
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.01,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbosity': 0,
    'n_jobs': -1,
}


# ============================================================================
# DATA LOADING (reuse from production)
# ============================================================================
def period_to_year(s):
    p = s.split('-')
    return (int(p[0]) + int(p[1])) / 2.0

def latlon_to_km_xy(lon, lat):
    return ((lon - BOUNDS[0]) * KM_PER_DEG_LON_HOUSTON,
            (lat - BOUNDS[2]) * KM_PER_DEG_LAT)

def load_wells(filepath, period):
    try:
        df = pd.read_csv(filepath, sep=r'\s+')
    except FileNotFoundError:
        return None
    for c in ['GWLm','Longitude','Latitude','well_constructed_depth','altitude']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['well_depth_m'] = df['well_constructed_depth'] * FT_TO_M
    df['altitude_m'] = df['altitude'] * FT_TO_M
    df['bottom_elevation_m'] = df['altitude_m'] - df['well_depth_m']
    df = df.dropna(subset=['Longitude','Latitude','GWLm','well_depth_m','altitude_m'])
    mask = ((df['Longitude'] >= BOUNDS[0]) & (df['Longitude'] <= BOUNDS[1]) &
            (df['Latitude'] >= BOUNDS[2]) & (df['Latitude'] <= BOUNDS[3]))
    df = df[mask].copy().reset_index(drop=True)
    df['period'] = period
    df['period_year'] = period_to_year(period)
    return df

def load_vhg(filepath, period):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return None
    mask = ((df['Longitude'] >= BOUNDS[0]) & (df['Longitude'] <= BOUNDS[1]) &
            (df['Latitude'] >= BOUNDS[2]) & (df['Latitude'] <= BOUNDS[3]))
    df = df[mask].copy().reset_index(drop=True)
    df['period'] = period
    df['period_year'] = period_to_year(period)
    return df

def add_neighborhood_features(wells_df, k=5, radius_km=10.0):
    df = wells_df.copy().reset_index(drop=True)
    x_km, y_km = latlon_to_km_xy(df['Longitude'].values, df['Latitude'].values)
    coords = np.column_stack([x_km, y_km])
    tree = cKDTree(coords)
    n = len(df)
    heads = df['GWLm'].values
    depths = df['bottom_elevation_m'].values
    min_idx = np.argmin(heads)
    local_head_mean = np.zeros(n)
    local_head_min = np.zeros(n)
    local_depth_mean = np.zeros(n)
    dist_to_min = np.zeros(n)
    for i in range(n):
        nq = min(k+1, n)
        _, idxs = tree.query(coords[i], k=nq)
        if nq > 1:
            ni = idxs[idxs != i][:k]
        else:
            ni = []
        if len(ni) > 0:
            local_head_mean[i] = np.mean(heads[ni])
            local_depth_mean[i] = np.mean(depths[ni])
        else:
            local_head_mean[i] = heads[i]
            local_depth_mean[i] = depths[i]
        ir = tree.query_ball_point(coords[i], radius_km)
        local_head_min[i] = np.min(heads[ir]) if ir else heads[i]
        dist_to_min[i] = np.sqrt((coords[i,0]-coords[min_idx,0])**2 +
                                  (coords[i,1]-coords[min_idx,1])**2)
    df['local_head_mean'] = local_head_mean
    df['local_head_min'] = local_head_min
    df['head_anomaly'] = df['GWLm'] - local_head_mean
    df['local_depth_mean'] = local_depth_mean
    df['dist_to_min_head'] = dist_to_min
    return df

def add_interaction_features(df):
    df = df.copy()
    df['Longitude_Latitude'] = df['Longitude'] * df['Latitude']
    df['GWLm_depth_ratio'] = df['GWLm'] / np.maximum(df['well_depth_m'].values, 10.0)
    return df


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 72)
    print(f"XGBOOST HYPERPARAMETER TUNING")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # Load and prepare data
    print("\nLoading data...")
    all_vhg = []
    for period in ALL_PERIODS:
        wells = load_wells(HEAD_FILE_PATTERN.format(period=period), period)
        if wells is None:
            continue
        wells = add_neighborhood_features(wells, k=N_NEAREST_WELLS,
                                           radius_km=LOCAL_RADIUS_KM)
        vhg = load_vhg(VHG_FILE_PATTERN.format(period=period), period)
        if vhg is None:
            continue
        vhg_ids = set(vhg['Well_ID'].values)
        matched = wells[wells['Well_ID'].isin(vhg_ids)].copy()
        matched['VHG_median'] = matched['Well_ID'].map(
            vhg.set_index('Well_ID')['VHG_median'].to_dict())
        matched = matched.dropna(subset=['VHG_median'])
        all_vhg.append(matched)
        print(f"  {period}: {len(matched)} wells")

    pooled = pd.concat(all_vhg, ignore_index=True)
    pooled = add_interaction_features(pooled)

    FEATURE_COLS = [
        'Longitude', 'Latitude', 'GWLm',
        'well_depth_m', 'altitude_m', 'bottom_elevation_m',
        'period_year', 'Longitude_Latitude', 'GWLm_depth_ratio',
        'local_head_mean', 'local_head_min', 'head_anomaly',
        'local_depth_mean', 'dist_to_min_head',
    ]

    X = pooled[FEATURE_COLS].copy()
    y = pooled['VHG_median'].values
    valid = X.notna().all(axis=1).values
    X = X.loc[valid].reset_index(drop=True)
    y = y[valid]

    # Target period mask
    periods = pooled.loc[valid, 'period'].values
    target_mask = periods == TARGET_PERIOD

    print(f"\n  Training samples: {len(X)}")
    print(f"  Target period samples: {target_mask.sum()}")

    # ================================================================
    # Test all parameter combinations
    # ================================================================
    depths = PARAM_GRID['max_depth']
    n_ests = PARAM_GRID['n_estimators']
    lrs = PARAM_GRID['learning_rate']
    mcws = PARAM_GRID['min_child_weight']

    total = len(depths) * len(n_ests) * len(lrs) * len(mcws)
    print(f"\n  Testing {total} parameter combinations...")
    print(f"  {'#':>3s}  {'depth':>5s}  {'n_est':>5s}  {'lr':>6s}  {'mcw':>4s}  "
          f"{'CV_R2':>8s}  {'CV_RMSE':>8s}  {'Tgt_R2':>8s}  {'Tgt_RMSE':>9s}")
    print("  " + "-" * 75)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    count = 0

    for depth, n_est, lr, mcw in itertools.product(depths, n_ests, lrs, mcws):
        count += 1
        params = {**FIXED_PARAMS, 'max_depth': depth,
                  'learning_rate': lr, 'min_child_weight': mcw}

        # 5-fold CV
        cv_r2 = []
        cv_rmse = []
        for tr_idx, vl_idx in kf.split(X):
            dtr = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx])
            dvl = xgb.DMatrix(X.iloc[vl_idx], label=y[vl_idx])
            mdl = xgb.train(params, dtr, num_boost_round=n_est,
                            verbose_eval=False)
            yp = mdl.predict(dvl)
            cv_r2.append(r2_score(y[vl_idx], yp))
            cv_rmse.append(np.sqrt(mean_squared_error(y[vl_idx], yp)))

        # Target period evaluation
        dtrain_full = xgb.DMatrix(X, label=y)
        model = xgb.train(params, dtrain_full, num_boost_round=n_est,
                          verbose_eval=False)

        if target_mask.sum() > 0:
            dt = xgb.DMatrix(X[target_mask])
            ytp = np.clip(model.predict(dt), VHG_MIN, VHG_MAX)
            tgt_r2 = r2_score(y[target_mask], ytp)
            tgt_rmse = np.sqrt(mean_squared_error(y[target_mask], ytp))
        else:
            tgt_r2 = np.nan
            tgt_rmse = np.nan

        r = {
            'max_depth': depth, 'n_estimators': n_est,
            'learning_rate': lr, 'min_child_weight': mcw,
            'cv_r2': np.mean(cv_r2), 'cv_r2_std': np.std(cv_r2),
            'cv_rmse': np.mean(cv_rmse),
            'tgt_r2': tgt_r2, 'tgt_rmse': tgt_rmse,
        }
        results.append(r)

        print(f"  {count:>3d}  {depth:>5d}  {n_est:>5d}  {lr:>6.3f}  {mcw:>4d}  "
              f"{np.mean(cv_r2):>8.4f}  {np.mean(cv_rmse):>8.4f}  "
              f"{tgt_r2:>8.4f}  {tgt_rmse:>9.4f}")

    # ================================================================
    # Rank results
    # ================================================================
    results_df = pd.DataFrame(results).sort_values('cv_r2', ascending=False)

    print(f"\n{'=' * 72}")
    print("TOP 10 CONFIGURATIONS (by CV R²)")
    print(f"{'=' * 72}")
    print(f"  {'Rank':>4s}  {'depth':>5s}  {'n_est':>5s}  {'lr':>6s}  {'mcw':>4s}  "
          f"{'CV_R2':>10s}  {'CV_RMSE':>8s}  {'Tgt_R2':>8s}")
    print("  " + "-" * 60)
    for i, (_, row) in enumerate(results_df.head(10).iterrows()):
        marker = " ★" if i == 0 else ""
        print(f"  {i+1:>4d}  {int(row['max_depth']):>5d}  "
              f"{int(row['n_estimators']):>5d}  {row['learning_rate']:>6.3f}  "
              f"{int(row['min_child_weight']):>4d}  "
              f"{row['cv_r2']:>7.4f}±{row['cv_r2_std']:.3f}  "
              f"{row['cv_rmse']:>8.4f}  {row['tgt_r2']:>8.4f}{marker}")

    # Current default for comparison
    print(f"\n  Current default: depth=3, n_est=500, lr=0.05, mcw=5")
    default = results_df[
        (results_df['max_depth'] == 3) &
        (results_df['n_estimators'] == 500) &
        (results_df['learning_rate'] == 0.05) &
        (results_df['min_child_weight'] == 5)
    ]
    if len(default) > 0:
        d = default.iloc[0]
        print(f"  Default CV R²:  {d['cv_r2']:.4f} ± {d['cv_r2_std']:.3f}")
        print(f"  Default Tgt R²: {d['tgt_r2']:.4f}")

    best = results_df.iloc[0]
    print(f"\n  Best CV R²:     {best['cv_r2']:.4f} ± {best['cv_r2_std']:.3f}")
    print(f"  Best Tgt R²:    {best['tgt_r2']:.4f}")
    print(f"  Best config:    depth={int(best['max_depth'])}, "
          f"n_est={int(best['n_estimators'])}, "
          f"lr={best['learning_rate']}, "
          f"mcw={int(best['min_child_weight'])}")

    improvement = best['cv_r2'] - (d['cv_r2'] if len(default) > 0 else 0)
    print(f"\n  Improvement: {improvement:+.4f} CV R²")
    if improvement < 0.02:
        print(f"  → Marginal improvement. Current defaults are near-optimal.")
    elif improvement < 0.05:
        print(f"  → Moderate improvement. Consider adopting the best config.")
    else:
        print(f"  → Significant improvement. Update production config.")

    # Save full results
    results_df.to_csv("XGBoost_tuning_results.csv", index=False,
                      float_format='%.4f')
    print(f"\n  Full results saved: XGBoost_tuning_results.csv")
    print("Done.")


if __name__ == "__main__":
    main()
