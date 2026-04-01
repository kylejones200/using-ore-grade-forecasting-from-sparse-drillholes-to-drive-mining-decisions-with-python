#!/usr/bin/env python3
"""
Ore Grade Forecasting - Production Implementation

Clean implementation of spatial grade estimation using Gaussian Processes
and geostatistical methods.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import cross_val_score, cross_val_predict
from scipy.spatial.distance import cdist
import time

# Import Tufte plotting utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tda_utils import setup_tufte_plot, TufteColors


def generate_synthetic_drillhole_data(num_holes=120, domain_size=1000, seed=42):
    """Generate realistic synthetic drillhole assay data."""
    np.random.seed(seed)
    x = np.random.uniform(0, domain_size, num_holes)
    y = np.random.uniform(0, domain_size, num_holes)
    z = np.random.uniform(-200, -50, num_holes)
    
    correlation_range = 150
    log_background = np.random.normal(-2, 0.5, num_holes)
    coords = np.column_stack([x, y, z])
    distances = cdist(coords, coords)
    correlation_matrix = np.exp(-(distances / correlation_range) ** 2)
    cholesky = np.linalg.cholesky(correlation_matrix + np.eye(num_holes) * 0.01)
    correlated_field = cholesky @ np.random.normal(0, 1, num_holes)
    log_grade = log_background + 0.8 * correlated_field
    grade_au_ppm = np.exp(log_grade)
    
    n_shoots = 3
    for _ in range(n_shoots):
        shoot_center = np.random.randint(0, num_holes)
        shoot_distances = np.linalg.norm(coords - coords[shoot_center], axis=1)
        shoot_influence = np.exp(-(shoot_distances / 80) ** 2)
        grade_au_ppm += shoot_influence * np.random.uniform(2, 8)
    
    return pd.DataFrame({
        'hole_id': [f'DH{i:03d}' for i in range(num_holes)],
        'x': x, 'y': y, 'z': z,
        'au_ppm': grade_au_ppm,
        'log_au_ppm': np.log(grade_au_ppm + 0.001)
    })

def calculate_experimental_variogram(data, max_distance=500, n_bins=20):
    """Calculate experimental variogram."""
    coords = data[['x', 'y', 'z']].values
    grades = data['log_au_ppm'].values
    n_samples = len(data)
    distances, semivariances = [], []
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= max_distance:
                semivar = 0.5 * (grades[i] - grades[j]) ** 2
                distances.append(dist)
                semivariances.append(semivar)
    
    distances = np.array(distances)
    semivariances = np.array(semivariances)
    bins = np.linspace(0, max_distance, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    binned_semivariance, bin_counts = [], []
    
    for i in range(n_bins):
        mask = (distances >= bins[i]) & (distances < bins[i + 1])
        binned_semivariance.append(semivariances[mask].mean() if mask.sum() > 0 else np.nan)
        bin_counts.append(mask.sum())
    
    binned_semivariance = np.array(binned_semivariance)
    valid_mask = ~np.isnan(binned_semivariance) & (np.array(bin_counts) >= 10)
    valid_distances = bin_centers[valid_mask]
    valid_semivar = binned_semivariance[valid_mask]
    
    if len(valid_semivar) >= 3:
        nugget = valid_semivar[0] if valid_semivar[0] < valid_semivar[-1] else 0
        sill = valid_semivar[-1]
        range_param = valid_distances[np.argmin(np.abs(valid_semivar - 0.95 * sill))]
    else:
        nugget, sill, range_param = 0, 1, 100
    
    return {'nugget': nugget, 'sill': sill, 'range': range_param}

def build_gp_grade_model(training_data):
    """Build Gaussian Process model for grade estimation."""
    X = training_data[['x', 'y', 'z']].values
    y = training_data['log_au_ppm'].values
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    
    kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(1.0, (0.1, 5.0)) + WhiteKernel(0.1, (0.01, 1.0))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, normalize_y=True)
    gp.fit(X_normalized, y)
    
    cv_scores = cross_val_score(gp, X_normalized, y, cv=5, scoring='r2')
    cv_predictions = cross_val_predict(gp, X_normalized, y, cv=5)
    mae = np.mean(np.abs(y - cv_predictions))
    rmse = np.sqrt(np.mean((y - cv_predictions) ** 2))
    
    return {'model': gp, 'X_mean': X_mean, 'X_std': X_std, 'cv_r2': cv_scores.mean(), 'cv_mae': mae, 'cv_rmse': rmse}

def estimate_block_model(drillhole_data, gp_model, block_size=25):
    """Estimate grades on 3D block model grid."""
    x_min, x_max = drillhole_data['x'].min(), drillhole_data['x'].max()
    y_min, y_max = drillhole_data['y'].min(), drillhole_data['y'].max()
    z_min, z_max = drillhole_data['z'].min(), drillhole_data['z'].max()
    
    x_blocks = np.arange(x_min, x_max, block_size)
    y_blocks = np.arange(y_min, y_max, block_size)
    z_blocks = np.arange(z_min, z_max, block_size)
    xx, yy, zz = np.meshgrid(x_blocks, y_blocks, z_blocks, indexing='ij')
    block_coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    block_coords_normalized = (block_coords - gp_model['X_mean']) / gp_model['X_std']
    
    mean_pred, std_pred = gp_model['model'].predict(block_coords_normalized, return_std=True)
    grade_mean = np.exp(mean_pred + std_pred ** 2 / 2)
    coefficient_of_variation = std_pred / np.abs(mean_pred)
    classification = np.where(coefficient_of_variation < 0.3, 'Measured',
                             np.where(coefficient_of_variation < 0.6, 'Indicated', 'Inferred'))
    
    return pd.DataFrame({
        'x': block_coords[:, 0], 'y': block_coords[:, 1], 'z': block_coords[:, 2],
        'au_ppm_mean': grade_mean, 'log_uncertainty': std_pred,
        'classification': classification, 'block_volume_m3': block_size ** 3
    })

def main():
    """Execute ore grade forecasting analysis."""
    print("=" * 70)
    print("ORE GRADE FORECASTING - PRODUCTION RUN")
    print("=" * 70)
    
    start_time = time.time()
    
    print("\n1. Generating Drillhole Data...")
    drillholes = generate_synthetic_drillhole_data(num_holes=120)
    print(f"   Generated {len(drillholes)} drillhole samples")
    print(f"   Grade range: {drillholes['au_ppm'].min():.3f} to {drillholes['au_ppm'].max():.2f} ppm Au")
    print(f"   Mean grade: {drillholes['au_ppm'].mean():.3f} ppm Au")
    
    print("\n2. Variogram Analysis...")
    variogram = calculate_experimental_variogram(drillholes)
    print(f"   Nugget: {variogram['nugget']:.3f}")
    print(f"   Sill: {variogram['sill']:.3f}")
    print(f"   Range: {variogram['range']:.1f} meters")
    
    print("\n3. Building Gaussian Process Model...")
    gp_model = build_gp_grade_model(drillholes)
    print(f"   Cross-Validated R²: {gp_model['cv_r2']:.3f}")
    print(f"   MAE: {gp_model['cv_mae']:.3f} log(ppm)")
    print(f"   RMSE: {gp_model['cv_rmse']:.3f} log(ppm)")
    
    print("\n4. Generating Block Model...")
    block_model = estimate_block_model(drillholes, gp_model, block_size=25)
    print(f"   Total Blocks: {len(block_model):,}")
    
    density_t_m3 = 2.7
    block_model['tonnage'] = block_model['block_volume_m3'] * density_t_m3
    cutoff_grade = 0.5
    ore_blocks = block_model[block_model['au_ppm_mean'] >= cutoff_grade]
    total_ore_tonnes = ore_blocks['tonnage'].sum()
    total_contained_gold = (ore_blocks['tonnage'] * ore_blocks['au_ppm_mean']).sum()
    average_ore_grade = total_contained_gold / total_ore_tonnes if total_ore_tonnes > 0 else 0
    
    print(f"   Ore Blocks (>{cutoff_grade} ppm): {len(ore_blocks):,}")
    print(f"   Total Ore Tonnage: {total_ore_tonnes:,.0f} tonnes")
    print(f"   Average Ore Grade: {average_ore_grade:.3f} ppm Au")
    print(f"   Contained Gold: {total_contained_gold / 1e6:.2f} million grams")
    
    print("\n5. Resource Classification...")
    resource_summary = ore_blocks.groupby('classification')['tonnage'].sum()
    for category in ['Measured', 'Indicated', 'Inferred']:
        tonnes = resource_summary.get(category, 0)
        pct = (tonnes / total_ore_tonnes * 100) if total_ore_tonnes > 0 else 0
        print(f"   {category}: {tonnes:,.0f} tonnes ({pct:.1f}%)")
    
    print("\n6. Exporting Results...")
    drillholes.to_csv('drillhole_data.csv', index=False)
    block_model.to_csv('block_model_grades.csv', index=False)
    print("   Exported: drillhole_data.csv")
    print("   Exported: block_model_grades.csv")
    
    execution_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    print(f"Total Execution Time: {execution_time:.3f} seconds")
    print(f"Drillholes Processed: {len(drillholes)}")
    print(f"Blocks Estimated: {len(block_model):,}")
    print(f"Model R²: {gp_model['cv_r2']:.3f}")
    print("=" * 70)

if __name__ == "__main__":
    main()

