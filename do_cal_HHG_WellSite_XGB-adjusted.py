#!/usr/bin/env python3
"""
HORIZONTAL GRADIENT ANALYSIS USING SITE-SPECIFIC VERTICAL GRADIENTS
===================================================================
Uses ML-predicted VHG values (from XGBoost) to adjust heads to a common reference
elevation (-300 m) before calculating horizontal gradients.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2, degrees
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
from pyproj import Transformer

# Houston region → UTM Zone 15N
transformer = Transformer.from_crs("epsg:4326", "epsg:32615", always_xy=True)

def project_to_km(lon, lat):
    x_m, y_m = transformer.transform(lon, lat)
    return x_m / 1000.0, y_m / 1000.0  # convert to km

def adjust_head_to_reference(head, bottom_elev, reference_elev=-300, vertical_gradient=None):
    """
    Adjust groundwater head to a reference elevation using site-specific vertical gradient.
    
    Parameters:
    -----------
    head : float
        Measured hydraulic head (m)
    bottom_elev : float
        Bottom elevation of well (m)
    reference_elev : float
        Target reference elevation (m), default -300
    vertical_gradient : float
        Site-specific VHG (m/m). If None, returns original head.
    
    Returns:
    --------
    adjusted_head : float
        Head adjusted to reference elevation
    """
    if vertical_gradient is None:
        return head
    
    delta_elev = reference_elev - bottom_elev
    delta_head = vertical_gradient * delta_elev
    adjusted_head = head + delta_head
    return adjusted_head

def calculate_condition_number(dx12, dy12, dx13, dy13):
    """
    Calculate condition number for triangle shape
    Condition number = 1 / |sin(angle)|
    Lower is better (well-distributed points)
    """
    # Lengths of vectors
    len1 = sqrt(dx12**2 + dy12**2)
    len2 = sqrt(dx13**2 + dy13**2)
    
    if len1 == 0 or len2 == 0:
        return np.inf
    
    # Normalize vectors
    ux1 = dx12 / len1
    uy1 = dy12 / len1
    ux2 = dx13 / len2
    uy2 = dy13 / len2
    
    # Determinant of normalized vectors = sin(angle)
    det_norm = abs(ux1 * uy2 - uy1 * ux2)
    
    if det_norm < 1e-10:
        return np.inf
    
    # Condition number = 1 / |sin(angle)|
    return 1.0 / det_norm

def calculate_aspect_ratio(d12, d13, d23):
    """Calculate triangle aspect ratio = longest_side / shortest_side"""
    sides = [d12, d13, d23]
    return max(sides) / min(sides)

def calculate_planar_gradient_conventional(p1, p2, p3, min_area_km2=None, max_area_km2=None):
    """
    True planar gradient in projected UTM coordinates (km)
    Returns flow direction (down-gradient)
    """

    lon1, lat1, h1 = p1
    lon2, lat2, h2 = p2
    lon3, lat3, h3 = p3

    # Project to true metric space (km)
    X1, Y1 = project_to_km(lon1, lat1)
    X2, Y2 = project_to_km(lon2, lat2)
    X3, Y3 = project_to_km(lon3, lat3)

    dx12 = X2 - X1
    dy12 = Y2 - Y1
    dh12 = h2 - h1

    dx13 = X3 - X1
    dy13 = Y3 - Y1
    dh13 = h3 - h1

    condition = calculate_condition_number(dx12, dy12, dx13, dy13)

    det = dx12 * dy13 - dy12 * dx13
    area_km2 = abs(det) / 2

    if min_area_km2 is not None and area_km2 < min_area_km2:
        return np.nan, np.nan, np.nan, np.nan, area_km2, condition

    if max_area_km2 is not None and area_km2 > max_area_km2:
        return np.nan, np.nan, np.nan, np.nan, area_km2, condition

    if abs(det) < 1e-12:
        return np.nan, np.nan, np.nan, np.nan, area_km2, condition

    # ∇h (up-gradient)
    grad_x = (dh12 * dy13 - dy12 * dh13) / det
    grad_y = (dx12 * dh13 - dh12 * dx13) / det

    # Flow direction (down-gradient)
    flow_x = -grad_x
    flow_y = -grad_y

    magnitude = sqrt(flow_x**2 + flow_y**2)

    # Bearing from north, clockwise (hydro standard)
    direction = (degrees(atan2(flow_x, flow_y)) + 360) % 360

    return flow_x, flow_y, magnitude, direction, area_km2, condition

def calculate_horizontal_gradient_3point(df, min_distance=5, max_distance=15, 
                                         min_area_km2=1.0, max_area_km2=200.0,
                                         max_aspect_ratio=5.0,
                                         max_gradient_magnitude=0.5, max_head_diff=50.0,
                                         max_condition=10.0):
    """
    Calculate horizontal gradients using 3-point method with site-specific adjusted heads.
    """
    results = []
    n_wells = len(df)
    total_triangles = n_wells * (n_wells - 1) * (n_wells - 2) // 6

    rejection_reasons = {
        'side_length': 0,
        'aspect_ratio': 0,
        'head_diff': 0,
        'area_too_small': 0,
        'area_too_large': 0,
        'gradient_magnitude': 0,
        'condition': 0,
        'collinear': 0
    }

    print(f"\nTotal possible triangles: {total_triangles:,}")
    print("\n  Projecting wells to UTM coordinates once...")

    # ------------------------------------------------------------
    # STEP 1 — Project all wells ONCE
    # ------------------------------------------------------------
    proj_coords = []
    for idx in range(n_wells):
        lon = df.iloc[idx]['Longitude']
        lat = df.iloc[idx]['Latitude']
        proj_coords.append(project_to_km(lon, lat))

    # ------------------------------------------------------------
    # STEP 2 — Precompute Euclidean distances in km
    # ------------------------------------------------------------
    print("  Pre-calculating Euclidean distances...")
    distances = {}
    for i in range(n_wells):
        x1, y1 = proj_coords[i]
        for j in range(i + 1, n_wells):
            x2, y2 = proj_coords[j]
            d = sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances[(i, j)] = d

    print("  Finding valid triangles...")
    triangle_count = 0

    # ------------------------------------------------------------
    # STEP 3 — Triangle search
    # ------------------------------------------------------------
    for i in range(n_wells):
        for j in range(i + 1, n_wells):
            for k in range(j + 1, n_wells):

                triangle_count += 1

                d12 = distances[(i, j)]
                d13 = distances[(i, k)]
                d23 = distances[(j, k)]

                # Constraint 1: Side lengths
                if not (min_distance <= d12 <= max_distance and
                        min_distance <= d13 <= max_distance and
                        min_distance <= d23 <= max_distance):
                    rejection_reasons['side_length'] += 1
                    continue

                # Constraint 2: Aspect ratio
                aspect_ratio = calculate_aspect_ratio(d12, d13, d23)
                if aspect_ratio > max_aspect_ratio:
                    rejection_reasons['aspect_ratio'] += 1
                    continue

                heads = [
                    df.iloc[i]['adjusted_head'],
                    df.iloc[j]['adjusted_head'],
                    df.iloc[k]['adjusted_head']
                ]

                # Constraint 3: Head difference
                if max(heads) - min(heads) > max_head_diff:
                    rejection_reasons['head_diff'] += 1
                    continue

                p1 = (df.iloc[i]['Longitude'], df.iloc[i]['Latitude'], heads[0])
                p2 = (df.iloc[j]['Longitude'], df.iloc[j]['Latitude'], heads[1])
                p3 = (df.iloc[k]['Longitude'], df.iloc[k]['Latitude'], heads[2])

                grad_x, grad_y, magnitude, direction, area, condition = \
                    calculate_planar_gradient_conventional(
                        p1, p2, p3,
                        min_area_km2=min_area_km2,
                        max_area_km2=max_area_km2
                    )

                if np.isnan(grad_x):
                    if area < min_area_km2:
                        rejection_reasons['area_too_small'] += 1
                    elif area > max_area_km2:
                        rejection_reasons['area_too_large'] += 1
                    else:
                        rejection_reasons['collinear'] += 1
                    continue

                # Constraint 4: Condition number
                if condition > max_condition:
                    rejection_reasons['condition'] += 1
                    continue

                # Constraint 5: Gradient magnitude
                if magnitude > max_gradient_magnitude:
                    rejection_reasons['gradient_magnitude'] += 1
                    continue

                center_lon = (p1[0] + p2[0] + p3[0]) / 3
                center_lat = (p1[1] + p2[1] + p3[1]) / 3

                results.append({
                    'center_lon': center_lon,
                    'center_lat': center_lat,
                    'well_1': df.iloc[i]['Well_ID'],
                    'well_2': df.iloc[j]['Well_ID'],
                    'well_3': df.iloc[k]['Well_ID'],
                    'gradient_x': grad_x,
                    'gradient_y': grad_y,
                    'gradient_magnitude': magnitude,
                    'gradient_direction': direction,
                    'side_12_km': d12,
                    'side_13_km': d13,
                    'side_23_km': d23,
                    'aspect_ratio': aspect_ratio,
                    'triangle_area_km2': area,
                    'condition_number': condition,
                    'head_1_adj': p1[2],
                    'head_2_adj': p2[2],
                    'head_3_adj': p3[2],
                    'vhg_1': df.iloc[i]['VHG_final'],
                    'vhg_2': df.iloc[j]['VHG_final'],
                    'vhg_3': df.iloc[k]['VHG_final']
                })

    print(f"\nACCEPTED triangles: {len(results):,} / {total_triangles:,}")
    return pd.DataFrame(results)


# ============================================================================
# MAIN PROGRAM
# ============================================================================

print("=" * 80)
print("HORIZONTAL GRADIENT ANALYSIS USING SITE-SPECIFIC VERTICAL GRADIENTS")
print("=" * 80)

# ============================================================================
# INPUT FILES
# ============================================================================

# Use ML-predicted VHG file (contains all well information including GWLm)
VHG_FILE = "VHG_ML_XGBoost_2020-2025.csv"

# Reference elevation (common datum for horizontal gradient calculation)
REFERENCE_ELEV = -300.0  # meters

# ============================================================================
# STEP 1: Load VHG data (contains GWLm, altitude_m, well_depth_m, VHG_final)
# ============================================================================

print(f"\nSTEP 1: Loading ML-predicted VHG data from: {VHG_FILE}")

df_vhg = pd.read_csv(VHG_FILE)

# Required columns check
required_cols = ['Well_ID', 'Longitude', 'Latitude', 'GWLm', 'altitude_m', 
                 'well_depth_m', 'VHG_final']
for col in required_cols:
    if col not in df_vhg.columns:
        print(f"ERROR: Required column '{col}' not found in {VHG_FILE}")
        exit(1)

# Convert to numeric
df_vhg['GWLm'] = pd.to_numeric(df_vhg['GWLm'], errors='coerce')
df_vhg['altitude_m'] = pd.to_numeric(df_vhg['altitude_m'], errors='coerce')
df_vhg['well_depth_m'] = pd.to_numeric(df_vhg['well_depth_m'], errors='coerce')
df_vhg['VHG_final'] = pd.to_numeric(df_vhg['VHG_final'], errors='coerce')
df_vhg['Longitude'] = pd.to_numeric(df_vhg['Longitude'], errors='coerce')
df_vhg['Latitude'] = pd.to_numeric(df_vhg['Latitude'], errors='coerce')

# Calculate bottom elevation
df_vhg['bottom_elevation_m'] = df_vhg['altitude_m'] - df_vhg['well_depth_m']

# Drop rows with missing critical data
df_clean = df_vhg[['Well_ID', 'GWLm', 'bottom_elevation_m', 'VHG_final', 
                    'Longitude', 'Latitude', 'altitude_m', 'well_depth_m']].copy()
df_clean = df_clean.dropna()

print(f"  Total wells loaded: {len(df_clean)}")
print(f"  VHG_final range: [{df_clean['VHG_final'].min():.4f}, {df_clean['VHG_final'].max():.4f}]")
print(f"  VHG_final mean: {df_clean['VHG_final'].mean():.4f}")
print(f"  VHG_final median: {df_clean['VHG_final'].median():.4f}")

# ============================================================================
# STEP 2: Adjust heads to reference elevation using site-specific VHG
# ============================================================================

print("\n" + "=" * 80)
print(f"STEP 2: Adjusting Heads to {REFERENCE_ELEV}m Reference Elevation")
print("=" * 80)

print(f"  Using site-specific VHG_final for each well")
print(f"  Reference elevation: {REFERENCE_ELEV} m")

# Apply site-specific adjustment
df_clean['adjusted_head'] = df_clean.apply(
    lambda row: adjust_head_to_reference(
        row['GWLm'], 
        row['bottom_elevation_m'], 
        REFERENCE_ELEV, 
        row['VHG_final']
    ), axis=1)

print(f"\n  Head adjustment summary:")
print(f"    Original head range: {df_clean['GWLm'].min():.2f} to {df_clean['GWLm'].max():.2f} m")
print(f"    Adjusted head range: {df_clean['adjusted_head'].min():.2f} to {df_clean['adjusted_head'].max():.2f} m")
print(f"    Mean adjustment: {(df_clean['adjusted_head'] - df_clean['GWLm']).mean():.2f} m")

# Statistics by VHG sign
vhg_positive = df_clean[df_clean['VHG_final'] > 0]
vhg_negative = df_clean[df_clean['VHG_final'] < 0]
print(f"\n  Wells with downward flow (VHG > 0): {len(vhg_positive)} ({100*len(vhg_positive)/len(df_clean):.1f}%)")
print(f"  Wells with upward flow (VHG < 0): {len(vhg_negative)} ({100*len(vhg_negative)/len(df_clean):.1f}%)")

# ============================================================================
# STEP 3: Calculate horizontal gradients
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Calculating Horizontal Gradients (Conventional 3-Point Method)")
print("=" * 80)

# Triangle constraints (adjusted for km units)
MIN_SIDE_KM = 2.0
MAX_SIDE_KM = 50.0
MIN_TRIANGLE_AREA_KM2 = 10.0
MAX_TRIANGLE_AREA_KM2 = 100.0
MAX_ASPECT_RATIO = 5.0
MAX_HEAD_DIFF = 25.0
MAX_GRADIENT_MAGNITUDE = 5.0
MAX_CONDITION = 5.0

print(f"\nTriangle constraints:")
print(f"  Minimum side length: {MIN_SIDE_KM} km")
print(f"  Maximum side length: {MAX_SIDE_KM} km")
print(f"  Minimum triangle area: {MIN_TRIANGLE_AREA_KM2} km²")
print(f"  Maximum triangle area: {MAX_TRIANGLE_AREA_KM2} km²")
print(f"  Maximum aspect ratio: {MAX_ASPECT_RATIO}")
print(f"  Maximum head difference: {MAX_HEAD_DIFF} m")
print(f"  Maximum gradient magnitude: {MAX_GRADIENT_MAGNITUDE} m/km")
print(f"  Maximum condition number: {MAX_CONDITION}")

# Calculate gradients
gradient_df = calculate_horizontal_gradient_3point(
    df_clean, 
    min_distance=MIN_SIDE_KM, 
    max_distance=MAX_SIDE_KM,
    min_area_km2=MIN_TRIANGLE_AREA_KM2,
    max_area_km2=MAX_TRIANGLE_AREA_KM2,
    max_aspect_ratio=MAX_ASPECT_RATIO,
    max_gradient_magnitude=MAX_GRADIENT_MAGNITUDE,
    max_head_diff=MAX_HEAD_DIFF,
    max_condition=MAX_CONDITION
)

if len(gradient_df) > 0:
    print(f"\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nFound {len(gradient_df)} valid triangles")
    
    # Statistics
    print(f"\nHorizontal Gradient Statistics (at triangle centroids):")
    print(f"  Mean magnitude: {gradient_df['gradient_magnitude'].mean():.4f} m/km")
    print(f"  Median magnitude: {gradient_df['gradient_magnitude'].median():.4f} m/km")
    print(f"  Std magnitude: {gradient_df['gradient_magnitude'].std():.4f} m/km")
    print(f"  Min magnitude: {gradient_df['gradient_magnitude'].min():.4f} m/km")
    print(f"  Max magnitude: {gradient_df['gradient_magnitude'].max():.4f} m/km")
    
    print(f"\nTriangle Geometry Statistics:")
    print(f"  Mean area: {gradient_df['triangle_area_km2'].mean():.2f} km²")
    print(f"  Median area: {gradient_df['triangle_area_km2'].median():.2f} km²")
    print(f"  Min area: {gradient_df['triangle_area_km2'].min():.2f} km²")
    print(f"  Max area: {gradient_df['triangle_area_km2'].max():.2f} km²")
    print(f"  Mean aspect ratio: {gradient_df['aspect_ratio'].mean():.2f}")
    print(f"  Median aspect ratio: {gradient_df['aspect_ratio'].median():.2f}")
    print(f"  Mean condition number: {gradient_df['condition_number'].mean():.2f}")
    print(f"  Median condition number: {gradient_df['condition_number'].median():.2f}")
    
    print(f"\nGradient Component Statistics:")
    print(f"  Gradient X (east-west, + = eastward):")
    print(f"    Mean: {gradient_df['gradient_x'].mean():.4f}, Median: {gradient_df['gradient_x'].median():.4f}")
    print(f"  Gradient Y (north-south, + = northward):")
    print(f"    Mean: {gradient_df['gradient_y'].mean():.4f}, Median: {gradient_df['gradient_y'].median():.4f}")
    
    print(f"\nVHG Statistics at triangle vertices:")
    print(f"  Mean VHG at vertices: {gradient_df[['vhg_1', 'vhg_2', 'vhg_3']].mean().mean():.4f}")
    print(f"  VHG range at vertices: {gradient_df[['vhg_1', 'vhg_2', 'vhg_3']].min().min():.4f} to {gradient_df[['vhg_1', 'vhg_2', 'vhg_3']].max().max():.4f}")
    
    # Save results
    output_file = 'Horizontal_Gradient_3Point_2020-2025.csv'
    gradient_df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\n✓ Results saved to: {output_file}")
    
    # Display first 10 results
    print("\nFirst 10 triangles:")
    display_cols = ['center_lon', 'center_lat', 'gradient_x', 'gradient_y', 
                    'gradient_magnitude', 'gradient_direction', 
                    'triangle_area_km2', 'aspect_ratio', 'condition_number']
    print(gradient_df[display_cols].head(10).to_string(index=False))
    
    # ============================================================================
    # STEP 4: Create visualizations
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Creating Visualizations")
    print("=" * 80)
    
    # Map with gradient vectors
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(gradient_df['center_lon'], gradient_df['center_lat'], 
                        c=gradient_df['gradient_magnitude'], cmap='viridis', 
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Horizontal Gradient Magnitude (m/km)', fontsize=10)
    
    # Plot quiver (arrows)
    step = max(1, len(gradient_df) // 50)
    ax.quiver(gradient_df['center_lon'][::step], gradient_df['center_lat'][::step],
              gradient_df['gradient_x'][::step], gradient_df['gradient_y'][::step],
              alpha=0.5, scale=300, width=0.003)
    
    ax.set_xlabel('Longitude (°W)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title(f'Horizontal Gradients at Triangle Centroids (2020-2025)\n'
                 f'Adjusted to {REFERENCE_ELEV}m using site-specific VHG (XGBoost)\n'
                 f'Constraints: sides {MIN_SIDE_KM}-{MAX_SIDE_KM} km, area {MIN_TRIANGLE_AREA_KM2}-{MAX_TRIANGLE_AREA_KM2} km²\n'
                 f'aspect ratio ≤{MAX_ASPECT_RATIO}, condition ≤{MAX_CONDITION}\n'
                 f'n = {len(gradient_df)} triangles', 
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Horizontal_Gradient_Map_2020-2025.png', dpi=150, bbox_inches='tight')
    print(f"✓ Map saved to: Horizontal_Gradient_Map_2020-2025.png")
    plt.close()
    
    # Histogram of magnitudes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(gradient_df['gradient_magnitude'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Horizontal Gradient Magnitude (m/km)', fontsize=12)
    ax.set_ylabel('Number of Triangles', fontsize=12)
    ax.set_title('Distribution of Horizontal Gradient Magnitudes (2020-2025)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Horizontal_Gradient_Histogram_2020-2025.png', dpi=150, bbox_inches='tight')
    print(f"✓ Histogram saved to: Horizontal_Gradient_Histogram_2020-2025.png")
    plt.close()
    
    # Area vs Magnitude
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(gradient_df['triangle_area_km2'], gradient_df['gradient_magnitude'], 
               alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    ax.set_xlabel('Triangle Area (km²)', fontsize=12)
    ax.set_ylabel('Gradient Magnitude (m/km)', fontsize=12)
    ax.set_title('Gradient Magnitude vs Triangle Area (2020-2025)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Gradient_vs_Area_2020-2025.png', dpi=150, bbox_inches='tight')
    print(f"✓ Area vs Magnitude plot saved to: Gradient_vs_Area_2020-2025.png")
    plt.close()
    
    # Condition number vs Magnitude
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(gradient_df['condition_number'], gradient_df['gradient_magnitude'], 
               alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    ax.set_xlabel('Condition Number', fontsize=12)
    ax.set_ylabel('Gradient Magnitude (m/km)', fontsize=12)
    ax.set_title('Gradient Magnitude vs Condition Number (2020-2025)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Gradient_vs_Condition_2020-2025.png', dpi=150, bbox_inches='tight')
    print(f"✓ Condition vs Magnitude plot saved to: Gradient_vs_Condition_2020-2025.png")
    plt.close()
    
    # NEW: VHG distribution map
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter_vhg = ax.scatter(df_clean['Longitude'], df_clean['Latitude'], 
                             c=df_clean['VHG_final'], cmap='RdYlBu_r', 
                             s=30, alpha=0.7, edgecolors='black', linewidth=0.3)
    cbar_vhg = plt.colorbar(scatter_vhg)
    cbar_vhg.set_label('Vertical Hydraulic Gradient (m/m)', fontsize=10)
    ax.set_xlabel('Longitude (°W)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title(f'Site-Specific VHG Values (2020-2025)\n'
                 f'From XGBoost Model (R² = 0.86)\n'
                 f'Blue = Upward Flow | Red = Downward Flow\n'
                 f'n = {len(df_clean)} wells', 
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('VHG_Distribution_Map_2020-2025.png', dpi=150, bbox_inches='tight')
    print(f"✓ VHG distribution map saved to: VHG_Distribution_Map_2020-2025.png")
    plt.close()
    
else:
    print("\nNo valid triangles found. Try adjusting constraints.")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETED")
print("=" * 80)
