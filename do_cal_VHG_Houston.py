#!/usr/bin/env python3
## 4-23-2026, confirmed Outlier detection and remove
## 4-16-20206. This is the final Cal_V program
"""
VERTICAL HYDRAULIC GRADIENT (VHG) CALCULATION + GRIDDING + OUTLIER FILTER
=========================================================================
Authors: Bob Wang (UH), with AI assistance
Date: April 2026

PURPOSE:
  Calculate pairwise vertical hydraulic gradients from wells at
  different depths within close horizontal proximity, remove spatial
  outliers using neighbor consensus, then grid to 5-km cells.

CHANGES FROM PREVIOUS VERSION (v2):
  1. MIN_DELTA_HEAD_M = 0.5 m (was 2.0 m) — less bias in low-VHG zones
     Reasoning: the 2.0 m filter systematically biased VHG upward in
     quiet areas by excluding legitimate small-head-difference pairs
     and leaving only pairs contaminated by pumping noise. The
     MIN_DELTA_ELEV_M = 50 m filter on the denominator already
     ensures each pair's gradient is physically meaningful.
  
  2. MAX_VHG = 0.20 (was 0.15) — don't over-clip at the source
     Reasoning: clipping at 0.15 truncates legitimate extreme values
     near pumping centers. Downstream outlier detection handles
     truly spurious extremes more selectively.
  
  3. Built-in outlier detection using MAD-based neighbor consensus
     Reasoning: identical method to the downstream ML program. For
     each well, compare its VHG_median to neighbors within 5 km.
     Flag as outlier if |VHG - neighbor_median| > max(0.05, 2.5×MAD).
     Outliers written to a separate file for inspection; cleaned
     output is the default input for downstream programs.

METHOD:
  For each well, find all wells within DISTANCE_THRESHOLD_KM.
  For each pair with:
    |delta_elev| > 50 m  (ensures denominator is meaningful)
    |delta_head| > 0.5 m (excludes only pure measurement noise)
    |VHG|         ≤ 0.20 (clips only physically implausible values)
  compute VHG = (head_i - head_j) / (bottom_elev_i - bottom_elev_j)

  The per-well representative VHG is the MEDIAN of all valid pairs.
  Per-well VHG values are then screened for spatial outliers before
  gridding to 5-km cells.

OUTPUT FILES:
  VHG_pairwise_{period}.csv       — per-well VHG, OUTLIERS REMOVED
  VHG_pairwise_all_{period}.csv   — per-well VHG, including flagged outliers
  VHG_gridded_{period}.csv        — 5-km gridded (from cleaned data)
  VHG_outliers_{period}.csv       — outliers only, for inspection
  VHG_summary_{period}.txt        — summary statistics
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from math import radians, sin, cos, sqrt, atan2
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION — CHANGE FOR EACH TIME PERIOD
# ============================================================================
data_period = "1970-1975"

HEAD_FILE = f"Mean_GWL_{data_period}.txt"
OUTPUT_PAIRWISE = f"VHG_pairwise_{data_period}.csv"              # cleaned (default)
OUTPUT_PAIRWISE_ALL = f"VHG_pairwise_all_{data_period}.csv"      # including outliers
OUTPUT_OUTLIERS = f"VHG_outliers_{data_period}.csv"              # outliers only
OUTPUT_GRIDDED = f"VHG_gridded_{data_period}.csv"
OUTPUT_SUMMARY = f"VHG_summary_{data_period}.txt"

# ============================================================================
# CONFIGURATION — FIXED
# ============================================================================

# Study area bounds
BOUNDS = [-96.0, -94.8, 29.0, 30.4]  # [lon_min, lon_max, lat_min, lat_max]

# Pairwise VHG calculation parameters
DISTANCE_THRESHOLD_KM = 2.0     # max horizontal distance between well pairs
MIN_DELTA_ELEV_M = 50.0         # min |vertical separation| between bottoms
MIN_DELTA_HEAD_M = 0.0          # min |head difference| (was 2.0)
MAX_VHG = 0.20                  # clip extreme VHG at the pair level

# Outlier detection (applied to per-well VHG_median values)
OUTLIER_N_NEIGHBORS = 5         # number of nearest wells to compare against
OUTLIER_RADIUS_KM = 5.0         # search radius for neighbors
OUTLIER_ABS_THRESHOLD = 0.05    # absolute floor: |VHG - median| > this
OUTLIER_MAD_MULTIPLIER = 2.5    # relative: > this × MAD
OUTLIER_MIN_NEIGHBORS = 3       # minimum neighbors required to evaluate

# Gridding parameters
GRID_SIZE_KM = 5.0
GRID_SIZE_DEG = GRID_SIZE_KM / 111.0
MIN_WELLS_PER_CELL = 1

# Unit conversion
FT_TO_M = 0.3048

# Local coordinate conversion (for outlier detection)
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON_HOUSTON = 111.0 * np.cos(np.radians(29.7))


# ============================================================================
# FUNCTIONS
# ============================================================================
def haversine_distance(lon1, lat1, lon2, lat2):
    """Great-circle distance in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def latlon_to_km_xy(lon, lat):
    """Convert lon/lat to local Cartesian km (for outlier detection KDTree)."""
    x_km = (lon - BOUNDS[0]) * KM_PER_DEG_LON_HOUSTON
    y_km = (lat - BOUNDS[2]) * KM_PER_DEG_LAT
    return x_km, y_km


def load_wells(filepath):
    """Load well data, convert units."""
    print(f"\n  Loading {filepath}")
    df = pd.read_csv(filepath, sep=r'\s+')

    df['GWLm'] = pd.to_numeric(df['GWLm'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['well_constructed_depth'] = pd.to_numeric(df['well_constructed_depth'], errors='coerce')
    df['altitude'] = pd.to_numeric(df['altitude'], errors='coerce')

    df['well_depth_m'] = df['well_constructed_depth'] * FT_TO_M
    df['altitude_m'] = df['altitude'] * FT_TO_M
    df['bottom_elevation_m'] = df['altitude_m'] - df['well_depth_m']

    df = df.dropna(subset=['Longitude', 'Latitude', 'GWLm', 'bottom_elevation_m'])

    # Filter to study area
    mask = ((df['Longitude'] >= BOUNDS[0]) & (df['Longitude'] <= BOUNDS[1]) &
            (df['Latitude'] >= BOUNDS[2]) & (df['Latitude'] <= BOUNDS[3]))
    df = df[mask].copy().reset_index(drop=True)

    print(f"  Wells in study area: {len(df)}")
    print(f"  GWLm range: [{df['GWLm'].min():.1f}, {df['GWLm'].max():.1f}] m")
    print(f"  Bottom elev range: [{df['bottom_elevation_m'].min():.0f}, "
          f"{df['bottom_elevation_m'].max():.0f}] m")

    # Bulk regression (diagnostic only)
    slope, intercept, r_value, _, _ = stats.linregress(
        df['bottom_elevation_m'].values, df['GWLm'].values)
    print(f"  Bulk regression: slope={slope:.4f}, R²={r_value**2:.4f} "
          f"(regional VHG estimate)")

    return df


def calculate_pairwise_vhg(df):
    """
    For each well, find nearby wells and compute pairwise VHG.
    Returns per-well median VHG with pair counts.
    """
    print(f"\n{'=' * 60}")
    print("Calculating Pairwise VHG")
    print(f"{'=' * 60}")
    print(f"  Distance threshold: {DISTANCE_THRESHOLD_KM} km")
    print(f"  Min |vertical separation|: {MIN_DELTA_ELEV_M} m")
    print(f"  Min |head difference|:     {MIN_DELTA_HEAD_M} m  (was 2.0)")
    print(f"  Max |VHG| (clip):          {MAX_VHG}  (was 0.15)")

    n_wells = len(df)
    results = []
    all_gradients = []
    wells_without_pairs = 0
    unique_pairs_counted = set()

    # KDTree in degrees with generous buffer
    deg_threshold = DISTANCE_THRESHOLD_KM / 111.0 * 1.5
    coords = df[['Longitude', 'Latitude']].values
    tree = cKDTree(coords)

    for i in range(n_wells):
        candidates = tree.query_ball_point(coords[i], deg_threshold)

        gradients = []
        nearby_count = 0

        for j in candidates:
            if i == j:
                continue

            # Precise distance check (haversine)
            dist = haversine_distance(
                df.loc[i, 'Longitude'], df.loc[i, 'Latitude'],
                df.loc[j, 'Longitude'], df.loc[j, 'Latitude'])

            if dist > DISTANCE_THRESHOLD_KM:
                continue

            nearby_count += 1

            delta_head = df.loc[i, 'GWLm'] - df.loc[j, 'GWLm']
            delta_elev = df.loc[i, 'bottom_elevation_m'] - df.loc[j, 'bottom_elevation_m']

            # Quality filters (ABSOLUTE values for both)
            if abs(delta_elev) > MIN_DELTA_ELEV_M and abs(delta_head) > MIN_DELTA_HEAD_M:
                gradient = delta_head / delta_elev
                if abs(gradient) <= MAX_VHG:
                    gradients.append(gradient)
                    # Count unique pairs only (avoid double-counting in stats)
                    pair_key = (min(i, j), max(i, j))
                    if pair_key not in unique_pairs_counted:
                        all_gradients.append(gradient)
                        unique_pairs_counted.add(pair_key)

        if len(gradients) > 0:
            median_vhg = np.median(gradients)
            results.append({
                'Well_ID': df.loc[i, 'Well_ID'],
                'Longitude': df.loc[i, 'Longitude'],
                'Latitude': df.loc[i, 'Latitude'],
                'altitude_m': df.loc[i, 'altitude_m'],
                'well_depth_m': df.loc[i, 'well_depth_m'],
                'bottom_elevation_m': df.loc[i, 'bottom_elevation_m'],
                'GWLm': df.loc[i, 'GWLm'],
                'VHG_median': median_vhg,
                'n_nearby_wells': nearby_count,
                'n_valid_pairs': len(gradients),
            })
        else:
            wells_without_pairs += 1

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Well_ID').reset_index(drop=True)

    # Statistics
    all_grad = np.array(all_gradients)
    print(f"\n  Results:")
    print(f"    Wells with valid VHG: {len(results_df)} / {n_wells}")
    print(f"    Wells without pairs:  {wells_without_pairs}")
    print(f"    Unique pairs:         {len(all_grad)}")

    if len(results_df) > 0:
        print(f"\n  Per-well median VHG:")
        print(f"    Mean:   {results_df['VHG_median'].mean():.4f} m/m")
        print(f"    Median: {results_df['VHG_median'].median():.4f} m/m")
        print(f"    Std:    {results_df['VHG_median'].std():.4f} m/m")
        print(f"    Range:  [{results_df['VHG_median'].min():.4f}, "
              f"{results_df['VHG_median'].max():.4f}]")

        n_down = (results_df['VHG_median'] > 0).sum()
        n_up = (results_df['VHG_median'] < 0).sum()
        print(f"    Downward (VHG>0): {n_down} ({100*n_down/len(results_df):.0f}%)")
        print(f"    Upward   (VHG<0): {n_up} ({100*n_up/len(results_df):.0f}%)")

        # Breakdown by n_valid_pairs
        print(f"\n  Breakdown by number of valid pairs per well:")
        print(f"    {'n_pairs':<10s} {'wells':>8s} {'mean_VHG':>10s} {'std_VHG':>10s}")
        for n_group in [1, 2, 3, 5, 10, 20]:
            if n_group == 20:
                mask = results_df['n_valid_pairs'] >= n_group
                label = f">={n_group}"
            else:
                next_n = [2, 3, 5, 10, 20][[1, 2, 3, 5, 10, 20].index(n_group)]
                mask = ((results_df['n_valid_pairs'] >= n_group) &
                        (results_df['n_valid_pairs'] < next_n))
                label = f"{n_group}-{next_n-1}" if next_n > n_group + 1 else f"{n_group}"
            n = mask.sum()
            if n > 0:
                m = results_df[mask]['VHG_median'].mean()
                s = results_df[mask]['VHG_median'].std()
                print(f"    {label:<10s} {n:>8d} {m:>10.4f} "
                      f"{s if not np.isnan(s) else 0:>10.4f}")

    if len(all_grad) > 0:
        print(f"\n  All unique pairwise VHG:")
        print(f"    Mean:   {all_grad.mean():.4f} m/m")
        print(f"    Median: {np.median(all_grad):.4f} m/m")
        print(f"    Count:  {len(all_grad)}")

    return results_df


def detect_spatial_outliers(pairwise_df):
    """
    Identify VHG outliers using spatial neighbor consensus (MAD-based).

    For each well's VHG_median:
      1. Find nearest OUTLIER_N_NEIGHBORS wells within OUTLIER_RADIUS_KM
      2. Compute median and MAD of those neighbors' VHG_median values
      3. Flag as outlier if:
           |VHG - neighbor_median| > max(OUTLIER_ABS_THRESHOLD,
                                        OUTLIER_MAD_MULTIPLIER × MAD)

    Notes:
      - Wells with fewer than OUTLIER_MIN_NEIGHBORS neighbors in radius
        are kept (insufficient basis to flag them as outliers).
      - Uses LOCAL Cartesian km coordinates (more accurate than degrees
        for a 5-km radius).

    Returns the input dataframe with added columns:
      is_outlier, neighbor_median, neighbor_mad, outlier_threshold
    """
    print(f"\n{'=' * 60}")
    print("Spatial Outlier Detection (per-well VHG_median)")
    print(f"{'=' * 60}")
    print(f"  Method: neighbor consensus with MAD")
    print(f"  Neighbors: {OUTLIER_N_NEIGHBORS} nearest within "
          f"{OUTLIER_RADIUS_KM} km")
    print(f"  Threshold: |VHG - median| > max("
          f"{OUTLIER_ABS_THRESHOLD}, {OUTLIER_MAD_MULTIPLIER} × MAD)")
    print(f"  Min neighbors required: {OUTLIER_MIN_NEIGHBORS}")

    df = pairwise_df.copy().reset_index(drop=True)
    df['is_outlier'] = False
    df['neighbor_median'] = np.nan
    df['neighbor_mad'] = np.nan
    df['outlier_threshold'] = np.nan

    if len(df) < OUTLIER_MIN_NEIGHBORS + 1:
        print(f"  Not enough wells for outlier detection — skipping")
        return df

    # Build KDTree in local km coordinates
    x_km, y_km = latlon_to_km_xy(df['Longitude'].values, df['Latitude'].values)
    coords = np.column_stack([x_km, y_km])
    tree = cKDTree(coords)
    vhgs = df['VHG_median'].values

    n_evaluated = 0
    n_insufficient = 0

    for i in range(len(df)):
        # Find wells within radius (exclude self)
        radius_idxs = tree.query_ball_point(coords[i], OUTLIER_RADIUS_KM)
        radius_idxs = [j for j in radius_idxs if j != i]

        if len(radius_idxs) < OUTLIER_MIN_NEIGHBORS:
            n_insufficient += 1
            continue

        # Take up to OUTLIER_N_NEIGHBORS closest
        if len(radius_idxs) > OUTLIER_N_NEIGHBORS:
            dists = np.linalg.norm(coords[radius_idxs] - coords[i], axis=1)
            order = np.argsort(dists)
            radius_idxs = [radius_idxs[k] for k in order[:OUTLIER_N_NEIGHBORS]]

        neighbor_vhgs = vhgs[radius_idxs]
        n_med = np.median(neighbor_vhgs)
        n_mad = np.median(np.abs(neighbor_vhgs - n_med))
        threshold = max(OUTLIER_ABS_THRESHOLD,
                        OUTLIER_MAD_MULTIPLIER * n_mad)

        df.at[i, 'neighbor_median'] = n_med
        df.at[i, 'neighbor_mad'] = n_mad
        df.at[i, 'outlier_threshold'] = threshold

        if abs(vhgs[i] - n_med) > threshold:
            df.at[i, 'is_outlier'] = True

        n_evaluated += 1

    n_outliers = df['is_outlier'].sum()

    print(f"\n  Outlier detection results:")
    print(f"    Wells evaluated: {n_evaluated}")
    print(f"    Wells with < {OUTLIER_MIN_NEIGHBORS} neighbors (kept): "
          f"{n_insufficient}")
    print(f"    Outliers flagged: {n_outliers} "
          f"({100*n_outliers/len(df):.1f}% of {len(df)} total)")

    # Show most extreme outliers
    if n_outliers > 0:
        outliers = df[df['is_outlier']].copy()
        outliers['deviation'] = outliers['VHG_median'] - outliers['neighbor_median']
        extreme = outliers.iloc[
            outliers['deviation'].abs().argsort()[::-1][:5]]
        print(f"\n  Top {min(5, len(extreme))} most extreme outliers:")
        print(f"    {'Well_ID':<18s} {'n_pairs':>8s} {'VHG':>9s} "
              f"{'NbrMed':>9s} {'Deviation':>11s}")
        for _, row in extreme.iterrows():
            print(f"    {str(row['Well_ID']):<18s} "
                  f"{int(row['n_valid_pairs']):>8d} "
                  f"{row['VHG_median']:>9.4f} "
                  f"{row['neighbor_median']:>9.4f} "
                  f"{row['deviation']:>+11.4f}")

    # Break down outliers by n_valid_pairs — test hypothesis that
    # sparse-pair wells are disproportionately flagged
    if n_outliers > 0:
        print(f"\n  Outliers by n_valid_pairs:")
        print(f"    {'n_pairs':<10s} {'total':>8s} {'outliers':>10s} {'% flagged':>12s}")
        for n_group, next_n in [(1, 2), (2, 3), (3, 5), (5, 10), (10, 9999)]:
            mask = ((df['n_valid_pairs'] >= n_group) &
                    (df['n_valid_pairs'] < next_n))
            n_total = mask.sum()
            if n_total == 0:
                continue
            n_out = df[mask]['is_outlier'].sum()
            label = f"{n_group}-{next_n-1}" if next_n != 9999 else f">={n_group}"
            pct = 100 * n_out / n_total if n_total > 0 else 0
            print(f"    {label:<10s} {n_total:>8d} {n_out:>10d} {pct:>11.1f}%")

    return df


def grid_vhg(pairwise_df):
    """Grid per-well VHG to 5-km cells using median."""
    print(f"\n{'=' * 60}")
    print("Gridding VHG to 5-km Cells")
    print(f"{'=' * 60}")

    lon = pairwise_df['Longitude'].values
    lat = pairwise_df['Latitude'].values
    vhg = pairwise_df['VHG_median'].values

    lon_edges = np.arange(BOUNDS[0], BOUNDS[1] + GRID_SIZE_DEG, GRID_SIZE_DEG)
    lat_edges = np.arange(BOUNDS[2], BOUNDS[3] + GRID_SIZE_DEG, GRID_SIZE_DEG)

    print(f"  Grid: {GRID_SIZE_KM} km cells")
    print(f"  Lon cells: {len(lon_edges)-1}, Lat cells: {len(lat_edges)-1}")

    lon_bin = np.digitize(lon, lon_edges) - 1
    lat_bin = np.digitize(lat, lat_edges) - 1

    results = []
    for xi in range(len(lon_edges) - 1):
        for yi in range(len(lat_edges) - 1):
            mask = (lon_bin == xi) & (lat_bin == yi)
            n_wells = mask.sum()

            if n_wells < MIN_WELLS_PER_CELL:
                continue

            cell_vhg = vhg[mask]
            lon_center = lon_edges[xi] + GRID_SIZE_DEG / 2
            lat_center = lat_edges[yi] + GRID_SIZE_DEG / 2

            results.append({
                'lon': lon_center,
                'lat': lat_center,
                'VHG_median': np.median(cell_vhg),
                'VHG_mean': np.mean(cell_vhg),
                'VHG_std': np.std(cell_vhg) if n_wells > 1 else 0.0,
                'n_wells': n_wells,
            })

    gridded = pd.DataFrame(results)

    print(f"  Grid cells with data: {len(gridded)}")
    if len(gridded) > 0:
        print(f"  VHG median: mean={gridded['VHG_median'].mean():.4f}, "
              f"median={gridded['VHG_median'].median():.4f} m/m")
        print(f"  Wells per cell: mean={gridded['n_wells'].mean():.1f}, "
              f"range=[{gridded['n_wells'].min()}, {gridded['n_wells'].max()}]")

    return gridded


def write_summary(df, pairwise_all, pairwise_clean, outliers, gridded_df,
                  filepath):
    """Write text summary of VHG analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"VERTICAL HYDRAULIC GRADIENT ANALYSIS: {data_period}")
    lines.append("=" * 70)
    lines.append(f"Input: {HEAD_FILE}")
    lines.append(f"Study area: {BOUNDS}")
    lines.append("")
    lines.append("PAIRWISE VHG PARAMETERS:")
    lines.append(f"  Distance threshold:          {DISTANCE_THRESHOLD_KM} km")
    lines.append(f"  Min |vertical separation|:   {MIN_DELTA_ELEV_M} m")
    lines.append(f"  Min |head difference|:       {MIN_DELTA_HEAD_M} m")
    lines.append(f"  Max |VHG| clip:              {MAX_VHG} m/m")
    lines.append("")
    lines.append("OUTLIER DETECTION PARAMETERS:")
    lines.append(f"  N nearest neighbors:         {OUTLIER_N_NEIGHBORS}")
    lines.append(f"  Search radius:               {OUTLIER_RADIUS_KM} km")
    lines.append(f"  Absolute threshold floor:    {OUTLIER_ABS_THRESHOLD} m/m")
    lines.append(f"  MAD multiplier:              {OUTLIER_MAD_MULTIPLIER}")
    lines.append(f"  Min neighbors to evaluate:   {OUTLIER_MIN_NEIGHBORS}")
    lines.append("")

    lines.append(f"RESULTS:")
    lines.append(f"  Wells loaded:                   {len(df)}")
    lines.append(f"  Wells with valid pairs:         {len(pairwise_all)}")
    lines.append(f"  Outliers removed:               {len(outliers)}")
    lines.append(f"  Final clean VHG observations:   {len(pairwise_clean)}")
    lines.append("")

    if len(pairwise_clean) > 0:
        lines.append("PER-WELL VHG (CLEANED):")
        lines.append(f"  Mean:   {pairwise_clean['VHG_median'].mean():.6f} m/m")
        lines.append(f"  Median: {pairwise_clean['VHG_median'].median():.6f} m/m")
        lines.append(f"  Std:    {pairwise_clean['VHG_median'].std():.6f} m/m")
        lines.append(f"  Min:    {pairwise_clean['VHG_median'].min():.6f} m/m")
        lines.append(f"  Max:    {pairwise_clean['VHG_median'].max():.6f} m/m")
        n_down = (pairwise_clean['VHG_median'] > 0).sum()
        n_up = (pairwise_clean['VHG_median'] < 0).sum()
        lines.append(f"  Downward: {n_down} ({100*n_down/len(pairwise_clean):.0f}%)")
        lines.append(f"  Upward:   {n_up} ({100*n_up/len(pairwise_clean):.0f}%)")
        lines.append("")

        lines.append("PERCENTILES (cleaned VHG_median):")
        for p in [5, 10, 25, 50, 75, 90, 95]:
            val = pairwise_clean['VHG_median'].quantile(p / 100)
            lines.append(f"  {p:3d}th: {val:.6f} m/m")
        lines.append("")

    if len(gridded_df) > 0:
        lines.append(f"GRIDDED VHG ({GRID_SIZE_KM} km cells):")
        lines.append(f"  Cells: {len(gridded_df)}")
        lines.append(f"  Median: {gridded_df['VHG_median'].median():.6f} m/m")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Summary: {filepath}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print(f"VERTICAL HYDRAULIC GRADIENT ANALYSIS v3: {data_period}")
    print("=" * 70)
    print(f"  Input: {HEAD_FILE}")
    print(f"  Area: {BOUNDS}")
    print(f"  Pair distance:  {DISTANCE_THRESHOLD_KM} km")
    print(f"  Min |d_elev|:   {MIN_DELTA_ELEV_M} m")
    print(f"  Min |d_head|:   {MIN_DELTA_HEAD_M} m  [reduced from 2.0]")
    print(f"  Max |VHG|:      {MAX_VHG}  [raised from 0.15]")
    print(f"  Outlier filter: MAD-based, enabled")

    # Step 1: Load wells
    print("\n" + "=" * 60)
    print("STEP 1: Loading Well Data")
    print("=" * 60)
    df = load_wells(HEAD_FILE)

    # Step 2: Pairwise VHG
    pairwise_all = calculate_pairwise_vhg(df)

    if len(pairwise_all) == 0:
        print("  No valid VHG pairs found. Exiting.")
        return

    # Step 3: Outlier detection
    pairwise_with_flags = detect_spatial_outliers(pairwise_all)

    outliers = pairwise_with_flags[pairwise_with_flags['is_outlier']].copy()
    pairwise_clean = pairwise_with_flags[~pairwise_with_flags['is_outlier']].copy()
    pairwise_clean = pairwise_clean.drop(
        columns=['is_outlier', 'neighbor_median', 'neighbor_mad',
                 'outlier_threshold'])

    # Step 4: Grid the cleaned data
    gridded_df = grid_vhg(pairwise_clean)

    # Step 5: Save outputs
    print(f"\n{'=' * 60}")
    print("Saving Outputs")
    print(f"{'=' * 60}")

    # Cleaned pairwise (default input for downstream programs)
    pairwise_clean.to_csv(OUTPUT_PAIRWISE, index=False, float_format='%.6f')
    print(f"  Cleaned pairwise: {OUTPUT_PAIRWISE} ({len(pairwise_clean)} wells)")

    # All pairwise including outliers (for reproducibility)
    pairwise_with_flags.to_csv(OUTPUT_PAIRWISE_ALL, index=False,
                                float_format='%.6f')
    print(f"  All pairwise:     {OUTPUT_PAIRWISE_ALL} "
          f"({len(pairwise_with_flags)} wells with is_outlier flag)")

    # Outliers only (for inspection)
    if len(outliers) > 0:
        outliers.to_csv(OUTPUT_OUTLIERS, index=False, float_format='%.6f')
        print(f"  Outliers:         {OUTPUT_OUTLIERS} ({len(outliers)} wells)")

    if len(gridded_df) > 0:
        gridded_df.to_csv(OUTPUT_GRIDDED, index=False, float_format='%.6f')
        print(f"  Gridded:          {OUTPUT_GRIDDED} ({len(gridded_df)} cells)")

    write_summary(df, pairwise_all, pairwise_clean, outliers, gridded_df,
                  OUTPUT_SUMMARY)

    # Final summary
    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"  Period: {data_period}")
    print(f"  Wells loaded:      {len(df)}")
    print(f"  Wells with VHG:    {len(pairwise_all)}")
    print(f"  Outliers removed:  {len(outliers)} "
          f"({100*len(outliers)/len(pairwise_all):.1f}%)")
    print(f"  Clean VHG output:  {len(pairwise_clean)} wells")
    print(f"  Grid cells:        {len(gridded_df)}")
    if len(pairwise_clean) > 0:
        print(f"  Clean VHG mean:    {pairwise_clean['VHG_median'].mean():.4f} m/m")
        print(f"  Clean VHG median:  {pairwise_clean['VHG_median'].median():.4f} m/m")


if __name__ == "__main__":
    main()
