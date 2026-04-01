#!/usr/bin/env python3
"""
Validation script for Ore Grade Forecasting blog code.
Tests all functions to ensure they run without errors.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.spatial.distance import cdist
from sklearn.model_selection import cross_val_score, cross_val_predict

# Import Tufte plotting utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tda_utils import setup_tufte_plot, TufteColors


def generate_synthetic_drillhole_data(num_holes=100, domain_size=1000, seed=42):
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
    
    drillholes = pd.DataFrame({
        'hole_id': [f'DH{i:03d}' for i in range(num_holes)],
        'x': x,
        'y': y,
        'z': z,
        'au_ppm': grade_au_ppm,
        'log_au_ppm': np.log(grade_au_ppm + 0.001)
    })
    
    return drillholes

def calculate_experimental_variogram(data, max_distance=500, n_bins=20):
    """Calculate experimental variogram to quantify spatial continuity."""
    coords = data[['x', 'y', 'z']].values
    grades = data['log_au_ppm'].values
    
    n_samples = len(data)
    distances = []
    semivariances = []
    
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
    binned_semivariance = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (distances >= bins[i]) & (distances < bins[i + 1])
        if mask.sum() > 0:
            binned_semivariance.append(semivariances[mask].mean())
            bin_counts.append(mask.sum())
        else:
            binned_semivariance.append(np.nan)
            bin_counts.append(0)
    
    binned_semivariance = np.array(binned_semivariance)
    
    valid_mask = ~np.isnan(binned_semivariance) & (np.array(bin_counts) >= 10)
    valid_distances = bin_centers[valid_mask]
    valid_semivar = binned_semivariance[valid_mask]
    
    # Pythonic variogram fitting
    if len(valid_semivar) >= 3:
        nugget = min(valid_semivar[0], valid_semivar[-1])
        sill = valid_semivar[-1]
        range_param = valid_distances[np.argmin(np.abs(valid_semivar - 0.95 * sill))]
    else:
        nugget, sill, range_param = 0, 1, 100
    
    return {
        'bin_centers': bin_centers,
        'binned_semivariance': binned_semivariance,
        'bin_counts': bin_counts,
        'nugget': nugget,
        'sill': sill,
        'range': range_param,
        'distances': distances,
        'semivariances': semivariances
    }

def build_gp_grade_model(training_data, kernel_params=None):
    """Build Gaussian Process model for grade estimation."""
    X = training_data[['x', 'y', 'z']].values
    y = training_data['log_au_ppm'].values
    
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    
    if kernel_params is None:
        length_scale = 1.0
        signal_variance = 1.0
        noise_variance = 0.1
    else:
        length_scale = kernel_params['length_scale']
        signal_variance = kernel_params['signal_variance']
        noise_variance = kernel_params['noise_variance']
    
    kernel = (
        ConstantKernel(signal_variance, (0.1, 10.0)) *
        RBF(length_scale=length_scale, length_scale_bounds=(0.1, 5.0)) +
        WhiteKernel(noise_level=noise_variance, noise_level_bounds=(0.01, 1.0))
    )
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=1e-6,
        normalize_y=True
    )
    
    gp.fit(X_normalized, y)
    
    cv_scores = cross_val_score(gp, X_normalized, y, cv=5, scoring='r2')
    cv_predictions = cross_val_predict(gp, X_normalized, y, cv=5)
    
    mae = np.mean(np.abs(y - cv_predictions))
    rmse = np.sqrt(np.mean((y - cv_predictions) ** 2))
    r2 = cv_scores.mean()
    
    return {
        'model': gp,
        'X_mean': X_mean,
        'X_std': X_std,
        'cv_r2': r2,
        'cv_mae': mae,
        'cv_rmse': rmse,
        'kernel_params': gp.kernel_,
        'log_marginal_likelihood': gp.log_marginal_likelihood()
    }

def estimate_block_model(drillhole_data, gp_model, block_size=25, domain_extent=None):
    """Estimate grades on 3D block model grid with uncertainty."""
    if domain_extent is None:
        x_min, x_max = drillhole_data['x'].min(), drillhole_data['x'].max()
        y_min, y_max = drillhole_data['y'].min(), drillhole_data['y'].max()
        z_min, z_max = drillhole_data['z'].min(), drillhole_data['z'].max()
    else:
        x_min, x_max = domain_extent['x']
        y_min, y_max = domain_extent['y']
        z_min, z_max = domain_extent['z']
    
    x_blocks = np.arange(x_min, x_max, block_size)
    y_blocks = np.arange(y_min, y_max, block_size)
    z_blocks = np.arange(z_min, z_max, block_size)
    
    xx, yy, zz = np.meshgrid(x_blocks, y_blocks, z_blocks, indexing='ij')
    block_coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    block_coords_normalized = (block_coords - gp_model['X_mean']) / gp_model['X_std']
    
    mean_pred, std_pred = gp_model['model'].predict(block_coords_normalized, return_std=True)
    
    grade_mean = np.exp(mean_pred + std_pred ** 2 / 2)
    grade_std = grade_mean * np.sqrt(np.exp(std_pred ** 2) - 1)
    
    grade_p10 = np.exp(mean_pred - 1.28 * std_pred)
    grade_p50 = np.exp(mean_pred)
    grade_p90 = np.exp(mean_pred + 1.28 * std_pred)
    
    # Pythonic classification with pd.cut
    coefficient_of_variation = std_pred / np.abs(mean_pred)
    classification = pd.cut(coefficient_of_variation,
                           bins=[0, 0.3, 0.6, np.inf],
                           labels=['Measured', 'Indicated', 'Inferred'])
    
    block_model = pd.DataFrame({
        'x': block_coords[:, 0],
        'y': block_coords[:, 1],
        'z': block_coords[:, 2],
        'au_ppm_mean': grade_mean,
        'au_ppm_std': grade_std,
        'au_ppm_p10': grade_p10,
        'au_ppm_p50': grade_p50,
        'au_ppm_p90': grade_p90,
        'log_uncertainty': std_pred,
        'classification': classification,
        'block_volume_m3': block_size ** 3
    })
    
    return block_model

def conditional_simulation(drillhole_data, gp_model, block_size=25, n_realizations=20):
    """Generate multiple equally-probable grade realizations."""
    x_blocks = np.arange(200, 800, block_size)
    y_blocks = np.arange(200, 800, block_size)
    z_blocks = np.arange(-150, -100, block_size)
    
    xx, yy, zz = np.meshgrid(x_blocks, y_blocks, z_blocks, indexing='ij')
    block_coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    block_coords_norm = (block_coords - gp_model['X_mean']) / gp_model['X_std']
    
    realizations = []
    
    for i in range(n_realizations):
        sample = gp_model['model'].sample_y(block_coords_norm, n_samples=1, random_state=i)
        grade_realization = np.exp(sample.ravel())
        realizations.append(grade_realization)
    
    realizations = np.array(realizations)
    
    mean_grade = realizations.mean(axis=0)
    std_grade = realizations.std(axis=0)
    p10_grade = np.percentile(realizations, 10, axis=0)
    p90_grade = np.percentile(realizations, 90, axis=0)
    
    global_means = realizations.mean(axis=1)
    global_p10 = np.percentile(global_means, 10)
    global_p50 = np.percentile(global_means, 50)
    global_p90 = np.percentile(global_means, 90)
    
    return {
        'realizations': realizations,
        'block_coords': block_coords,
        'mean_grade': mean_grade,
        'std_grade': std_grade,
        'p10_grade': p10_grade,
        'p90_grade': p90_grade,
        'global_p10': global_p10,
        'global_p50': global_p50,
        'global_p90': global_p90
    }

def main():
    """Run validation tests."""
    print("=" * 70)
    print("ORE GRADE FORECASTING - CODE VALIDATION")
    print("=" * 70)
    
    np.random.seed(42)
    
    print("\n1. Testing drillhole data generation...")
    drillholes = generate_synthetic_drillhole_data(num_holes=120)
    print(f"   ✓ Generated {len(drillholes)} drillhole samples")
    print(f"   ✓ Grade range: {drillholes['au_ppm'].min():.3f} to {drillholes['au_ppm'].max():.2f} ppm Au")
    print(f"   ✓ Mean grade: {drillholes['au_ppm'].mean():.3f} ppm Au")
    
    print("\n2. Testing variogram analysis...")
    variogram = calculate_experimental_variogram(drillholes)
    print(f"   ✓ Nugget Effect: {variogram['nugget']:.3f}")
    print(f"   ✓ Sill: {variogram['sill']:.3f}")
    print(f"   ✓ Range: {variogram['range']:.1f} meters")
    
    print("\n3. Testing Gaussian Process model...")
    gp_model = build_gp_grade_model(drillholes)
    print(f"   ✓ Cross-Validated R²: {gp_model['cv_r2']:.3f}")
    print(f"   ✓ MAE: {gp_model['cv_mae']:.3f} log(ppm)")
    print(f"   ✓ RMSE: {gp_model['cv_rmse']:.3f} log(ppm)")
    
    print("\n4. Testing block model estimation...")
    block_model = estimate_block_model(drillholes, gp_model, block_size=50)
    density_t_m3 = 2.7
    block_model['tonnage'] = block_model['block_volume_m3'] * density_t_m3
    
    cutoff_grade = 0.5
    ore_blocks = block_model[block_model['au_ppm_mean'] >= cutoff_grade]
    
    total_ore_tonnes = ore_blocks['tonnage'].sum()
    total_contained_gold = (ore_blocks['tonnage'] * ore_blocks['au_ppm_mean']).sum()
    average_ore_grade = total_contained_gold / total_ore_tonnes if total_ore_tonnes > 0 else 0
    
    print(f"   ✓ Total Blocks: {len(block_model):,}")
    print(f"   ✓ Ore Blocks: {len(ore_blocks):,}")
    print(f"   ✓ Average Ore Grade: {average_ore_grade:.3f} ppm Au")
    
    print("\n5. Testing conditional simulation...")
    simulations = conditional_simulation(drillholes, gp_model, n_realizations=10)
    print(f"   ✓ Realizations Generated: 10")
    print(f"   ✓ Blocks per Realization: {len(simulations['mean_grade']):,}")
    print(f"   ✓ Global P50: {simulations['global_p50']:.3f} ppm Au")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)

if __name__ == "__main__":
    main()

