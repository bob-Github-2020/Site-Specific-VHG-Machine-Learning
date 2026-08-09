#!/usr/bin/env python3
## 4-16-2026, 14 features, no weighting period
## 4-16-2026 - v3: Final polished version, diagnostics added, bugs fixed
## 4-15-2026 - v2: Added neighborhood features, outlier removal, uniform weighting
## 4-13-2026
## 4-9-2026. XGBoost is the final Choice. Great.

"""
MULTI-PERIOD ML VHG PREDICTION - XGBOOST v3 (FINAL)
====================================================
Pipeline position: Stage 2 of a two-stage pipeline.

  Stage 1 (upstream, do_cal_VHG_Houston.py):
    - Pairwise VHG calculation
    - MAD-based outlier removal at per-well level
    - Outputs VHG_pairwise_{period}.csv  <- input to this program

  Stage 2 (THIS PROGRAM):
    - Multi-period pooling with neighborhood features
    - XGBoost regression with 14 features
    - Predicts VHG at all monitoring wells for the target period

KEY CHOICES:
  1. NEIGHBORHOOD FEATURES (5 new + 9 original = 14 total):
     - local_head_mean  : mean GWLm of 5 nearest wells
     - local_head_min   : minimum GWLm within 10 km
     - head_anomaly     : GWLm - local_head_mean (relative stress)
     - local_depth_mean : mean bottom_elevation of 5 nearest wells
     - dist_to_min_head : distance to lowest-head well

  2. UNIFORM WEIGHTING (no temporal decay):
     All periods contribute equally. The period_year feature lets the
     model learn temporal patterns. This avoids throwing away the
     data-rich 1970s/1980s periods (which decay=0.5 would down-weight
     to ~6% of their information).

  3. OUTLIER FILTER OFF by default:
     Upstream cleaning in Stage 1 is sufficient. Setting this ON
     applies a second (redundant) filter and typically removes very
     few additional samples. Keep OFF for simpler pipeline.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
import warnings
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
TARGET_PERIOD = "2020-2025"

# VHG physical constraints (applied to ML predictions)
VHG_MIN = -0.15
VHG_MAX = 0.15
WARN_IF_CLIPPING = True

# Neighborhood feature parameters
N_NEAREST_WELLS = 5      # for local_head_mean, local_depth_mean
LOCAL_RADIUS_KM = 10.0   # for local_head_min

# Outlier detection (default OFF — upstream cleaning is sufficient)
APPLY_OUTLIER_FILTER = False   # True=apply filter; False=skip
OUTLIER_N_NEIGHBORS = 5
OUTLIER_RADIUS_KM = 5.0
OUTLIER_ABS_THRESHOLD = 0.05
OUTLIER_MAD_MULTIPLIER = 2.5

ALL_PERIODS = [
    "1970-1975",
    "1980-1985",
    "1990-1995",
    "2000-2005",
    "2010-2015",
    "2020-2025",
]

HEAD_FILE_PATTERN = "Mean_GWL_{period}.txt"
VHG_FILE_PATTERN = "VHG_pairwise_{period}.csv"

OUTPUT_FILE = f"VHG_ML_XGBoost_{TARGET_PERIOD}.csv"
SUMMARY_FILE = f"XGBoost_VHG_summary_{TARGET_PERIOD}.txt"

# HGSD polygons
HGSD_AREA12_FILE = "HGSD_Area1and2_Wang.xy"
HGSD_AREA3_FILE = "HGSD_Area3.xy"

BOUNDS = [-96.0, -94.8, 29.0, 30.4]
FT_TO_M = 0.3048

# Local Cartesian conversion (for distance computations)
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON_HOUSTON = 111.0 * np.cos(np.radians(29.7))  # ~96.4 km

# XGBoost hyperparameters
XGB_PARAMS = {
    'n_estimators': 500,  # 500,
    'max_depth': 4,   # 3
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,  # 5
    'reg_alpha': 0.01,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbosity': 0,
    'n_jobs': -1,
}

# Diagnostic toggle
PRINT_CORRELATION_MATRIX = True   # compute pairwise feature correlations


class Tee:
    """Duplicate output to screen and log file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ============================================================================
# DATA LOADING
# ============================================================================
def period_to_year(period_str):
    """Return midpoint year of a period like '1980-1985' -> 1982.5"""
    parts = period_str.split('-')
    return (int(parts[0]) + int(parts[1])) / 2.0


def latlon_to_km_xy(lon, lat):
    """Convert lon/lat to local Cartesian km."""
    x_km = (lon - BOUNDS[0]) * KM_PER_DEG_LON_HOUSTON
    y_km = (lat - BOUNDS[2]) * KM_PER_DEG_LAT
    return x_km, y_km


def load_wells(filepath, period):
    """Load wells with derived columns. Returns None on missing file."""
    try:
        df = pd.read_csv(filepath, sep=r'\s+')
    except FileNotFoundError:
        return None

    for col in ['GWLm', 'Longitude', 'Latitude',
                'well_constructed_depth', 'altitude']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['well_depth_m'] = df['well_constructed_depth'] * FT_TO_M
    df['altitude_m'] = df['altitude'] * FT_TO_M
    df['bottom_elevation_m'] = df['altitude_m'] - df['well_depth_m']

    df = df.dropna(subset=['Longitude', 'Latitude', 'GWLm',
                           'well_depth_m', 'altitude_m'])

    mask = ((df['Longitude'] >= BOUNDS[0]) & (df['Longitude'] <= BOUNDS[1]) &
            (df['Latitude'] >= BOUNDS[2]) & (df['Latitude'] <= BOUNDS[3]))
    df = df[mask].copy().reset_index(drop=True)

    df['period'] = period
    df['period_year'] = period_to_year(period)
    return df


def load_vhg(filepath, period):
    """Load pairwise VHG observations."""
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


def read_hgsd_polygon(filepath):
    """Read a single polygon from a lon/lat text file."""
    coords = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('>'):
                    parts = line.split()
                    if len(parts) >= 2:
                        coords.append((float(parts[0]), float(parts[1])))
    except FileNotFoundError:
        return None
    return Polygon(coords) if len(coords) >= 3 else None


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def add_neighborhood_features(wells_df, k=5, radius_km=10.0):
    """
    Add neighborhood features per period.

    Uses ALL wells in the period (not just those with VHG) to compute
    neighborhood statistics — so the spatial context is the same
    whether the well is a training sample or a prediction target.

    Features added:
      local_head_mean   : mean GWLm of k nearest wells (excluding self)
      local_head_min    : minimum GWLm within radius_km
      head_anomaly      : GWLm - local_head_mean
      local_depth_mean  : mean bottom_elevation_m of k nearest wells
      dist_to_min_head  : distance (km) to lowest-head well in this period
    """
    df = wells_df.copy().reset_index(drop=True)

    x_km, y_km = latlon_to_km_xy(df['Longitude'].values, df['Latitude'].values)
    coords = np.column_stack([x_km, y_km])
    tree = cKDTree(coords)

    n = len(df)
    local_head_mean = np.zeros(n)
    local_head_min = np.zeros(n)
    local_depth_mean = np.zeros(n)
    dist_to_min_head = np.zeros(n)

    heads = df['GWLm'].values
    depths = df['bottom_elevation_m'].values

    min_head_idx = np.argmin(heads)
    min_head_x, min_head_y = coords[min_head_idx]

    for i in range(n):
        n_query = min(k + 1, n)
        _, idxs = tree.query(coords[i], k=n_query)

        if n_query > 1:
            mask = idxs != i
            neighbor_idxs = idxs[mask][:k]
        else:
            neighbor_idxs = []

        if len(neighbor_idxs) > 0:
            local_head_mean[i] = np.mean(heads[neighbor_idxs])
            local_depth_mean[i] = np.mean(depths[neighbor_idxs])
        else:
            local_head_mean[i] = heads[i]
            local_depth_mean[i] = depths[i]

        idxs_in_radius = tree.query_ball_point(coords[i], radius_km)
        if len(idxs_in_radius) > 0:
            local_head_min[i] = np.min(heads[idxs_in_radius])
        else:
            local_head_min[i] = heads[i]

        dist_to_min_head[i] = np.sqrt(
            (coords[i, 0] - min_head_x) ** 2 +
            (coords[i, 1] - min_head_y) ** 2)

    df['local_head_mean'] = local_head_mean
    df['local_head_min'] = local_head_min
    df['head_anomaly'] = df['GWLm'] - df['local_head_mean']
    df['local_depth_mean'] = local_depth_mean
    df['dist_to_min_head'] = dist_to_min_head

    return df


def add_interaction_features(df):
    """Cross-product and ratio features."""
    df = df.copy()
    df['Longitude_Latitude'] = df['Longitude'] * df['Latitude']
    # Use max(well_depth, 10) to avoid extreme ratios for very shallow wells
    safe_depth = np.maximum(df['well_depth_m'].values, 10.0)
    df['GWLm_depth_ratio'] = df['GWLm'].values / safe_depth
    return df


# ============================================================================
# OUTLIER DETECTION (optional, usually done upstream)
# ============================================================================
def detect_vhg_outliers(vhg_df, n_neighbors=5, radius_km=5.0,
                        abs_threshold=0.05, mad_multiplier=2.5):
    """
    MAD-based neighbor consensus outlier detection (per period).

    For each well: find n_neighbors nearest within radius_km in the SAME
    period. Compute neighbor median and MAD. Flag if:
        |VHG - median| > max(abs_threshold, mad_multiplier × MAD)
    """
    df = vhg_df.copy().reset_index(drop=True)
    df['is_outlier'] = False
    df['neighbor_median'] = np.nan
    df['neighbor_mad'] = np.nan
    df['outlier_threshold'] = np.nan

    for period in df['period'].unique():
        mask = df['period'] == period
        sub = df[mask].copy()
        if len(sub) < 4:
            continue

        x_km, y_km = latlon_to_km_xy(sub['Longitude'].values,
                                      sub['Latitude'].values)
        coords = np.column_stack([x_km, y_km])
        tree = cKDTree(coords)
        vhgs = sub['VHG_median'].values
        sub_idx = sub.index.values

        for local_i, i in enumerate(sub_idx):
            radius_idxs = tree.query_ball_point(coords[local_i], radius_km)
            radius_idxs = [j for j in radius_idxs if j != local_i]

            if len(radius_idxs) < 3:
                continue

            if len(radius_idxs) > n_neighbors:
                dists = np.linalg.norm(
                    coords[radius_idxs] - coords[local_i], axis=1)
                order = np.argsort(dists)
                radius_idxs = [radius_idxs[k] for k in order[:n_neighbors]]

            neighbor_vhgs = vhgs[radius_idxs]
            n_med = np.median(neighbor_vhgs)
            n_mad = np.median(np.abs(neighbor_vhgs - n_med))
            threshold = max(abs_threshold, mad_multiplier * n_mad)

            df.at[i, 'is_outlier'] = abs(vhgs[local_i] - n_med) > threshold
            df.at[i, 'neighbor_median'] = n_med
            df.at[i, 'neighbor_mad'] = n_mad
            df.at[i, 'outlier_threshold'] = threshold

    return df


def apply_vhg_constraint(vhg_values, min_val, max_val, warn=False):
    """Clip predictions to physical range."""
    orig_min = vhg_values.min()
    orig_max = vhg_values.max()
    clipped = np.clip(vhg_values, min_val, max_val)

    if warn:
        n_lo = int((vhg_values < min_val).sum())
        n_hi = int((vhg_values > max_val).sum())
        if n_lo > 0 or n_hi > 0:
            print(f"    WARNING: {n_lo} clipped to {min_val}, "
                  f"{n_hi} clipped to {max_val}")
            print(f"             Original range: [{orig_min:.4f}, {orig_max:.4f}]")
    return clipped


# ============================================================================
# DIAGNOSTICS
# ============================================================================
def print_correlation_matrix(df, features):
    """Print correlation matrix for diagnostic purposes."""
    print(f"\n  Feature correlation matrix (|r| > 0.7 flagged with *):")
    corr = df[features].corr()

    # Header with truncated feature names
    print(f"    {'':22s}", end='')
    for f in features:
        print(f"{f[:8]:>9s}", end='')
    print()

    for r_feat in features:
        print(f"    {r_feat:<22s}", end='')
        for c_feat in features:
            val = corr.loc[r_feat, c_feat]
            marker = '*' if abs(val) > 0.7 and r_feat != c_feat else ' '
            print(f"{val:>+8.2f}{marker}", end='')
        print()

    # Redundant pairs
    print(f"\n  Pairs with |r| > 0.7 (indicates redundancy):")
    seen = set()
    redundant_count = 0
    for r_feat in features:
        for c_feat in features:
            if r_feat != c_feat and abs(corr.loc[r_feat, c_feat]) > 0.7:
                pair = tuple(sorted([r_feat, c_feat]))
                if pair not in seen:
                    seen.add(pair)
                    print(f"    {r_feat:<22s} ↔ {c_feat:<22s}  "
                          f"r = {corr.loc[r_feat, c_feat]:+.3f}")
                    redundant_count += 1
    if redundant_count == 0:
        print(f"    (none — all features are independent)")


def print_vhg_distribution(df, label):
    """Print summary statistics of VHG values."""
    v = df['VHG_median']
    print(f"  {label}:")
    print(f"    Mean:   {v.mean():.4f}")
    print(f"    Median: {v.median():.4f}")
    print(f"    Std:    {v.std():.4f}")
    print(f"    Range:  [{v.min():.4f}, {v.max():.4f}]")
    n_pos = (v > 0).sum()
    n_neg = (v < 0).sum()
    print(f"    Downward (VHG>0): {n_pos} ({100*n_pos/len(v):.1f}%)")
    print(f"    Upward   (VHG<0): {n_neg} ({100*n_neg/len(v):.1f}%)")


# ============================================================================
# MAIN
# ============================================================================
def main():
    tee = Tee(SUMMARY_FILE)
    sys.stdout = tee

    # Initialize to avoid NameError if filter is off
    n_outliers_total = 0

    print("=" * 70)
    print(f"XGBOOST VHG PREDICTION v3 (FINAL)")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"Target period:      {TARGET_PERIOD}")
    print(f"Sample weighting:   UNIFORM (period_year feature for time info)")
    print(f"Outlier filter:     {APPLY_OUTLIER_FILTER}")
    if APPLY_OUTLIER_FILTER:
        print(f"  threshold: max({OUTLIER_ABS_THRESHOLD}, "
              f"{OUTLIER_MAD_MULTIPLIER} × MAD), within {OUTLIER_RADIUS_KM} km, "
              f"{OUTLIER_N_NEIGHBORS} neighbors")
    else:
        print(f"  (upstream cleaning in do_cal_VHG_Houston.py is sufficient)")
    print(f"Neighborhood feats: k={N_NEAREST_WELLS}, "
          f"radius={LOCAL_RADIUS_KM} km")
    print(f"VHG prediction clip: [{VHG_MIN}, {VHG_MAX}]")
    print("=" * 70)

    area12 = read_hgsd_polygon(HGSD_AREA12_FILE)
    area3 = read_hgsd_polygon(HGSD_AREA3_FILE)

    # ================================================================
    # STEP 1: Load all periods, compute neighborhood features
    # ================================================================
    print(f"\nSTEP 1: Loading All Periods + Neighborhood Features")

    all_vhg_list = []
    all_wells_target = None

    for period in ALL_PERIODS:
        print(f"\n  Period: {period}")

        wells = load_wells(HEAD_FILE_PATTERN.format(period=period), period)
        if wells is None:
            print(f"    WARNING: head file missing, skipping")
            continue
        print(f"    Wells: {len(wells)}")

        # Neighborhood features computed on ALL wells in this period
        wells = add_neighborhood_features(
            wells, k=N_NEAREST_WELLS, radius_km=LOCAL_RADIUS_KM)

        if period == TARGET_PERIOD:
            all_wells_target = wells.copy()

        vhg = load_vhg(VHG_FILE_PATTERN.format(period=period), period)
        if vhg is None:
            print(f"    WARNING: VHG file missing, skipping")
            continue
        print(f"    VHG observations: {len(vhg)}")

        # Match VHG to wells with attributes
        vhg_ids = set(vhg['Well_ID'].values)
        matched = wells[wells['Well_ID'].isin(vhg_ids)].copy()
        vhg_lookup = vhg.set_index('Well_ID')['VHG_median'].to_dict()
        matched['VHG_median'] = matched['Well_ID'].map(vhg_lookup)
        matched = matched.dropna(subset=['VHG_median'])
        matched['sample_weight'] = 1.0  # UNIFORM

        # Also carry n_valid_pairs if it exists in the VHG file
        if 'n_valid_pairs' in vhg.columns:
            pairs_lookup = vhg.set_index('Well_ID')['n_valid_pairs'].to_dict()
            matched['n_valid_pairs'] = matched['Well_ID'].map(pairs_lookup)

        print(f"    Matched (well+VHG): {len(matched)}")
        all_vhg_list.append(matched)

    if all_wells_target is None:
        print("\nERROR: Target period wells not loaded. Exiting.")
        tee.close()
        return

    pooled = pd.concat(all_vhg_list, ignore_index=True)
    print(f"\n  POOLED TRAINING DATA: {len(pooled)} samples")
    for period in ALL_PERIODS:
        n = (pooled['period'] == period).sum()
        if n > 0:
            print(f"    {period}: {n:>4d} samples")

    # Distribution diagnostic
    print()
    print_vhg_distribution(pooled, "Pooled VHG distribution (all periods)")

    # n_valid_pairs diagnostic (if available)
    if 'n_valid_pairs' in pooled.columns:
        npairs = pooled['n_valid_pairs']
        print(f"\n  VHG reliability by n_valid_pairs:")
        print(f"    {'n_pairs':<10s} {'count':>8s} {'VHG_mean':>10s} "
              f"{'VHG_std':>10s}")
        for lo, hi, label in [(1, 2, '1'), (2, 3, '2'),
                               (3, 5, '3-4'), (5, 10, '5-9'),
                               (10, 9999, '>=10')]:
            mask = (npairs >= lo) & (npairs < hi)
            n = mask.sum()
            if n > 0:
                m = pooled.loc[mask, 'VHG_median'].mean()
                s = pooled.loc[mask, 'VHG_median'].std() if n > 1 else 0
                print(f"    {label:<10s} {n:>8d} {m:>10.4f} {s:>10.4f}")

    # ================================================================
    # STEP 2: Outlier detection (optional — usually upstream already)
    # ================================================================
    print(f"\nSTEP 2: Outlier Detection")

    if APPLY_OUTLIER_FILTER:
        pooled = detect_vhg_outliers(
            pooled,
            n_neighbors=OUTLIER_N_NEIGHBORS,
            radius_km=OUTLIER_RADIUS_KM,
            abs_threshold=OUTLIER_ABS_THRESHOLD,
            mad_multiplier=OUTLIER_MAD_MULTIPLIER)

        n_outliers_total = int(pooled['is_outlier'].sum())
        n_evaluated = int(pooled['neighbor_median'].notna().sum())
        n_no_neighbors = len(pooled) - n_evaluated

        print(f"  Wells evaluated: {n_evaluated}")
        print(f"  Wells with < 3 neighbors (kept): {n_no_neighbors}")
        print(f"  Outliers flagged: {n_outliers_total} "
              f"({100*n_outliers_total/len(pooled):.1f}%)")

        print(f"\n  Outliers by period:")
        for period in ALL_PERIODS:
            mask = pooled['period'] == period
            n_total = int(mask.sum())
            if n_total == 0:
                continue
            n_out = int(pooled[mask]['is_outlier'].sum())
            print(f"    {period}: {n_out:>3d} / {n_total:>3d} "
                  f"({100*n_out/n_total:.1f}%)")

        outliers = pooled[pooled['is_outlier']].copy()
        if len(outliers) > 0:
            outliers['deviation'] = (outliers['VHG_median'] -
                                     outliers['neighbor_median'])
            top = outliers.iloc[
                outliers['deviation'].abs().argsort()[::-1][:5]]
            print(f"\n  Top {len(top)} most extreme outliers:")
            print(f"    {'Well_ID':<18s} {'Period':<12s} "
                  f"{'VHG':>8s} {'NbrMed':>9s} {'Deviation':>11s}")
            for _, row in top.iterrows():
                print(f"    {str(row['Well_ID']):<18s} "
                      f"{row['period']:<12s} "
                      f"{row['VHG_median']:>+8.4f} "
                      f"{row['neighbor_median']:>+9.4f} "
                      f"{row['deviation']:>+11.4f}")

        pooled = pooled[~pooled['is_outlier']].copy()
        print(f"\n  Training set after outlier removal: {len(pooled)}")
    else:
        print(f"  Filter DISABLED (upstream cleaning provides input data)")

    # ================================================================
    # STEP 3: Feature engineering & diagnostics
    # ================================================================
    print(f"\nSTEP 3: Feature Engineering")

    pooled = add_interaction_features(pooled)
    all_wells_target = add_interaction_features(all_wells_target)

    FEATURE_COLS = [
        # Original location/state features (9)
        'Longitude', 'Latitude', 'GWLm',
        'well_depth_m', 'altitude_m', 'bottom_elevation_m',
        'period_year',
        'Longitude_Latitude', 'GWLm_depth_ratio',
        # Neighborhood features (5)
        'local_head_mean', 'local_head_min', 'head_anomaly',
        'local_depth_mean', 'dist_to_min_head',
    ]

    X_train = pooled[FEATURE_COLS].copy()
    y_train = pooled['VHG_median'].values
    sample_weights = pooled['sample_weight'].values

    # Handle NaN consistently (convert to numpy for uniform indexing)
    valid = X_train.notna().all(axis=1).values
    n_dropped = int((~valid).sum())
    X_train = X_train.loc[valid].reset_index(drop=True)
    y_train = y_train[valid]
    sample_weights = sample_weights[valid]

    print(f"  Training samples: {len(X_train)}"
          + (f" (dropped {n_dropped} with missing features)" if n_dropped else ""))
    print(f"  Total features: {len(FEATURE_COLS)} "
          f"(9 original + 5 neighborhood)")

    # Feature correlation diagnostic
    if PRINT_CORRELATION_MATRIX:
        pooled_valid = pooled.loc[valid].reset_index(drop=True)
        print_correlation_matrix(pooled_valid, FEATURE_COLS)

    # ================================================================
    # STEP 4: Train XGBoost
    # ================================================================
    print(f"\nSTEP 4: Training XGBoost Model")

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)

    model = xgb.train(
        XGB_PARAMS, dtrain,
        num_boost_round=XGB_PARAMS['n_estimators'],
        verbose_eval=False)

    # 5-fold CV
    print(f"\n  5-fold cross-validation:")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores = []
    cv_rmse_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_vl = y_train[train_idx], y_train[val_idx]
        w_tr = sample_weights[train_idx]

        dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dvl = xgb.DMatrix(X_vl, label=y_vl)
        fold_model = xgb.train(
            XGB_PARAMS, dtr,
            num_boost_round=XGB_PARAMS['n_estimators'],
            verbose_eval=False)
        y_pred = fold_model.predict(dvl)
        cv_r2_scores.append(r2_score(y_vl, y_pred))
        cv_rmse_scores.append(np.sqrt(mean_squared_error(y_vl, y_pred)))

    cv_r2_mean = float(np.mean(cv_r2_scores))
    cv_r2_std = float(np.std(cv_r2_scores))
    cv_rmse_mean = float(np.mean(cv_rmse_scores))
    cv_rmse_std = float(np.std(cv_rmse_scores))

    print(f"    CV R²:   {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    print(f"    CV RMSE: {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f} m/m")

    # Weighted training performance
    y_train_pred = model.predict(dtrain)
    train_r2 = r2_score(y_train, y_train_pred, sample_weight=sample_weights)
    print(f"    Train R² (weighted): {train_r2:.4f}")

    # Feature importances
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False)
    total_imp = importance_df['importance'].sum()
    importance_df['pct'] = 100 * importance_df['importance'] / total_imp

    NEW_FEATS = {'local_head_mean', 'local_head_min', 'head_anomaly',
                 'local_depth_mean', 'dist_to_min_head'}

    print(f"\n  Feature importances (by gain):")
    for _, row in importance_df.iterrows():
        marker = " ← neighborhood" if row['feature'] in NEW_FEATS else ""
        print(f"    {row['feature']:<22s} {row['importance']:>10.4f} "
              f"({row['pct']:>5.1f}%){marker}")

    # ================================================================
    # STEP 5: Evaluate on target period
    # ================================================================
    print(f"\nSTEP 5: Evaluate on Target Period ({TARGET_PERIOD})")

    # Use same pooled_valid for consistency
    pooled_valid = pooled.loc[valid].reset_index(drop=True) if not APPLY_OUTLIER_FILTER \
                   else pooled.reset_index(drop=True)
    # Simpler: just rebuild target subset from X_train + period info
    periods_valid = pooled.loc[valid, 'period'].values if not APPLY_OUTLIER_FILTER \
                    else pooled['period'].values
    target_mask_in_train = periods_valid == TARGET_PERIOD

    r2_target = None
    rmse_target = None
    if target_mask_in_train.sum() > 0:
        X_target = X_train[target_mask_in_train]
        y_target = y_train[target_mask_in_train]
        dtarget = xgb.DMatrix(X_target)
        y_target_pred = apply_vhg_constraint(
            model.predict(dtarget), VHG_MIN, VHG_MAX, warn=WARN_IF_CLIPPING)

        r2_target = r2_score(y_target, y_target_pred)
        rmse_target = np.sqrt(mean_squared_error(y_target, y_target_pred))
        mae_target = mean_absolute_error(y_target, y_target_pred)
        corr_target = np.corrcoef(y_target, y_target_pred)[0, 1]

        print(f"\n  Target period ({len(y_target)} pts):")
        print(f"    R²:    {r2_target:.4f}")
        print(f"    RMSE:  {rmse_target:.4f} m/m")
        print(f"    MAE:   {mae_target:.4f} m/m")
        print(f"    Corr:  {corr_target:.4f}")
        print(f"    Obs mean:  {y_target.mean():.4f}, "
              f"Pred mean: {y_target_pred.mean():.4f}")

        diff = y_target_pred - y_target
        within = int((np.abs(diff) <= 0.01).sum())
        print(f"\n  Difference statistics (pred - obs):")
        print(f"    Mean:   {diff.mean():+.6f}")
        print(f"    Median: {np.median(diff):+.6f}")
        print(f"    Std:    {diff.std():.6f}")
        print(f"    Within ±0.01: {within} ({100*within/len(y_target):.1f}%)")
    else:
        print(f"  WARNING: no target-period samples in training set")

    # ================================================================
    # STEP 6: Predict for all target-period wells
    # ================================================================
    print(f"\nSTEP 6: Predicting for All {TARGET_PERIOD} Wells")

    X_all = all_wells_target[FEATURE_COLS].copy()
    dall = xgb.DMatrix(X_all)
    vhg_predicted_raw = model.predict(dall)
    vhg_predicted = apply_vhg_constraint(
        vhg_predicted_raw, VHG_MIN, VHG_MAX, warn=WARN_IF_CLIPPING)

    result = all_wells_target[['Well_ID', 'Longitude', 'Latitude',
                                'GWLm', 'altitude_m', 'well_depth_m',
                                'bottom_elevation_m']].copy()
    result['VHG_predicted_raw'] = vhg_predicted_raw
    result['VHG_predicted'] = vhg_predicted
    result['period'] = TARGET_PERIOD

    # Merge observed VHG (from CLEANED upstream file)
    target_vhg = load_vhg(VHG_FILE_PATTERN.format(period=TARGET_PERIOD),
                          TARGET_PERIOD)
    if target_vhg is not None:
        vhg_lookup = target_vhg.set_index('Well_ID')['VHG_median'].to_dict()
        result['VHG_observed'] = result['Well_ID'].map(vhg_lookup)
        result['has_observed_VHG'] = result['VHG_observed'].notna()
    else:
        result['VHG_observed'] = np.nan
        result['has_observed_VHG'] = False

    # Final: use observed where available, predicted elsewhere
    result['VHG_final'] = np.where(
        result['has_observed_VHG'],
        result['VHG_observed'],
        result['VHG_predicted'])

    n_obs = int(result['has_observed_VHG'].sum())
    n_pred = len(result) - n_obs
    print(f"  Total wells: {len(result)}")
    print(f"    With observed VHG: {n_obs}")
    print(f"    ML-predicted only: {n_pred}")

    # Per-zone summary
    print(f"\n  Per-Zone Summary (VHG_final):")
    for zone_name, poly in [('Area 3       ', area3),
                            ('Area 1&2     ', area12)]:
        if poly is None:
            continue
        points = [Point(lo, la) for lo, la in zip(result['Longitude'],
                                                   result['Latitude'])]
        zmask = np.array([poly.contains(p) for p in points])
        if zmask.sum() == 0:
            continue
        zone = result[zmask]
        print(f"    {zone_name} ({int(zmask.sum()):>3d} wells): "
              f"mean={zone['VHG_final'].mean():+.4f}, "
              f"median={zone['VHG_final'].median():+.4f}, "
              f"std={zone['VHG_final'].std():.4f}")

    # Outside zones
    points = [Point(lo, la) for lo, la in zip(result['Longitude'],
                                               result['Latitude'])]
    outside_mask = np.ones(len(result), dtype=bool)
    for poly in [area12, area3]:
        if poly is None:
            continue
        inside = np.array([poly.contains(p) for p in points])
        outside_mask = outside_mask & (~inside)
    if outside_mask.sum() > 0:
        outside = result[outside_mask]
        print(f"    Outside      ({int(outside_mask.sum()):>3d} wells): "
              f"mean={outside['VHG_final'].mean():+.4f}, "
              f"median={outside['VHG_final'].median():+.4f}, "
              f"std={outside['VHG_final'].std():.4f}")

    # ================================================================
    # STEP 7: Save outputs
    # ================================================================
    output_cols = ['Well_ID', 'Longitude', 'Latitude', 'GWLm',
                   'altitude_m', 'well_depth_m', 'bottom_elevation_m',
                   'VHG_predicted_raw', 'VHG_predicted', 'VHG_observed',
                   'VHG_final', 'has_observed_VHG', 'period']
    result[output_cols].to_csv(OUTPUT_FILE, index=False, float_format='%.6f')
    print(f"\n  Saved: {OUTPUT_FILE}")

    # Final summary box
    print(f"\n{'=' * 60}")
    print(f"FINAL RESULTS — XGBoost v3")
    print(f"{'=' * 60}")
    print(f"  Target period:       {TARGET_PERIOD}")
    print(f"  Sample weighting:    UNIFORM")
    print(f"  Outlier filter:      {'ON' if APPLY_OUTLIER_FILTER else 'OFF (upstream)'}")
    if APPLY_OUTLIER_FILTER:
        print(f"  Outliers removed:    {n_outliers_total}")
    print(f"  Training samples:    {len(X_train)}")
    print(f"  Total features:      {len(FEATURE_COLS)} "
          f"(9 original + 5 neighborhood)")
    print(f"  VHG prediction clip: [{VHG_MIN}, {VHG_MAX}]")
    print(f"")
    print(f"  CV R²:               {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    print(f"  CV RMSE:             {cv_rmse_mean:.4f} m/m")
    if r2_target is not None:
        print(f"  Target R²:           {r2_target:.4f}")
        print(f"  Target RMSE:         {rmse_target:.4f} m/m")
    print(f"  Training R² (wtd):   {train_r2:.4f}")
    print(f"")
    print(f"  Output CSV:   {OUTPUT_FILE}")
    print(f"  Summary log:  {SUMMARY_FILE}")
    print(f"{'=' * 60}")

    tee.close()
    sys.stdout = tee.terminal


if __name__ == "__main__":
    main()
