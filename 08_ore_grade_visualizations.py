#!/usr/bin/env python3
import sys
import os

# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS

"""
Generate visualizations for Ore Grade Forecasting blog post.
Uses minimalist styling with serif fonts, clean axes, and high-quality output.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.spatial.distance import cdist
from sklearn.model_selection import cross_val_predict
import sys
import os

import sys
import os

# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS

# Import Tufte plotting utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tda_utils import setup_tufte_plot, TufteColors



def save_fig(filename):
    """Save plot in the standard minimalist format."""
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

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
    
    return pd.DataFrame({
        'x': x, 'y': y, 'z': z,
        'au_ppm': grade_au_ppm,
        'log_au_ppm': np.log(grade_au_ppm + 0.001)
    })

def calculate_experimental_variogram(data, max_distance=500, n_bins=20):
    """Calculate experimental variogram."""
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
    
    return bin_centers, np.array(binned_semivariance), np.array(bin_counts)

def build_simple_gp_model(training_data):
    """Build simplified GP model for visualization."""
    X = training_data[['x', 'y', 'z']].values
    y = training_data['log_au_ppm'].values
    
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    
    kernel = (ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1))
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                   alpha=1e-6, normalize_y=True)
    gp.fit(X_normalized, y)
    
    return gp, X_mean, X_std, X_normalized, y

def create_main_visualization():
    """Create main ore grade spatial analysis visualization."""
    np.random.seed(42)
    
    # Generate data
    drillholes = generate_synthetic_drillhole_data(num_holes=120)
    
    # Create figure with three panels
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    # Panel 1: Drillhole locations colored by grade
    scatter = ax1.scatter(drillholes['x'], drillholes['y'], 
                         c=drillholes['au_ppm'], s=50,
                         cmap='Greys', vmin=0, vmax=drillholes['au_ppm'].quantile(0.95),
                         edgecolors='black', linewidths=0.5)
    
    # Add high-grade highlights (hollow circles)
    high_grade = drillholes[drillholes['au_ppm'] > drillholes['au_ppm'].quantile(0.90)]
    ax1.scatter(high_grade['x'], high_grade['y'], s=100,
                facecolors='none', edgecolors='black', linewidths=2, zorder=5)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Au Grade (ppm)', fontsize=10)
    cbar.outline.set_visible(False)
    
    # Apply minimalist style
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_position(("outward", 5))
    ax1.spines["bottom"].set_position(("outward", 5))
    ax1.set_title('Drillhole Spatial Distribution', fontsize=12, fontweight="bold", loc="left")
    ax1.set_xlabel('Easting (m)', fontsize=10)
    ax1.set_ylabel('Northing (m)', fontsize=10)
    ax1.set_aspect('equal')
    
    # Panel 2: Experimental variogram
    bin_centers, binned_semivar, bin_counts = calculate_experimental_variogram(drillholes)
    
    # Plot points with size proportional to pair count
    valid_mask = ~np.isnan(binned_semivar) & (bin_counts >= 10)
    valid_distances = bin_centers[valid_mask]
    valid_semivar = binned_semivar[valid_mask]
    valid_counts = bin_counts[valid_mask]
    
    # Scale marker sizes
    marker_sizes = 50 + (valid_counts / valid_counts.max()) * 150
    
    ax2.scatter(valid_distances, valid_semivar, s=marker_sizes,
                color='white', edgecolors='black', linewidths=1.5, zorder=5)
    
    # Fit spherical model
    if len(valid_semivar) >= 3:
        nugget = min(valid_semivar[0], valid_semivar[-1])
        sill = valid_semivar[-1]
        range_param = valid_distances[np.argmin(np.abs(valid_semivar - 0.95 * sill))]
        
        # Plot model
        h = np.linspace(0, 500, 100)
        gamma = np.where(h < range_param,
                        nugget + (sill - nugget) * (1.5 * (h / range_param) - 0.5 * (h / range_param) ** 3),
                        sill)
        ax2.plot(h, gamma, 'k-', linewidth=1.5, label=f'Spherical Model (range={range_param:.0f}m)')
        
        # Mark parameters
        ax2.axhline(y=sill, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.axhline(y=nugget, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax2.axvline(x=range_param, color='gray', linestyle='-.', linewidth=0.8, alpha=0.5)
        
        ax2.text(10, sill * 1.05, f'Sill = {sill:.2f}', fontsize=8)
        ax2.text(10, nugget * 0.5, f'Nugget = {nugget:.2f}', fontsize=8)
    
    # Apply minimalist style
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_position(("outward", 5))
    ax2.spines["bottom"].set_position(("outward", 5))
    ax2.set_title('Experimental Variogram', fontsize=12, fontweight="bold", loc="left")
    ax2.set_xlabel('Separation Distance (m)', fontsize=10)
    ax2.set_ylabel('Semivariance', fontsize=10)
    ax2.legend(loc='lower right', frameon=False, fontsize=9)
    ax2.set_xlim(0, 500)
    
    # Panel 3: Grade distribution (histogram)
    grades = drillholes['au_ppm']
    
    # Create bins
    bins = np.linspace(0, grades.quantile(0.95), 20)
    counts, edges = np.histogram(grades, bins=bins)
    bin_centers_hist = (edges[:-1] + edges[1:]) / 2
    
    ax3.bar(bin_centers_hist, counts, width=np.diff(edges)[0] * 0.9,
            color='white', edgecolor='black', linewidth=1.5)
    
    # Add statistics
    mean_grade = grades.mean()
    median_grade = grades.median()
    
    ax3.axvline(x=mean_grade, color='black', linestyle='--', linewidth=1.5, label=f'Mean = {mean_grade:.2f} ppm')
    ax3.axvline(x=median_grade, color='gray', linestyle='-.', linewidth=1.5, label=f'Median = {median_grade:.2f} ppm')
    
    # Apply minimalist style
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_position(("outward", 5))
    ax3.spines["bottom"].set_position(("outward", 5))
    ax3.set_title('Grade Distribution', fontsize=12, fontweight="bold", loc="left")
    ax3.set_xlabel('Au Grade (ppm)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.legend(loc='upper right', frameon=False, fontsize=9)
    
    # Save
    save_fig('08_ore_grade_main.png')
    print("✓ Created: 08_ore_grade_main.png")

def create_accuracy_visualization():
    """Create GP model accuracy visualization."""
    np.random.seed(42)
    
    # Generate data and build model
    drillholes = generate_synthetic_drillhole_data(num_holes=120)
    gp, X_mean, X_std, X_normalized, y = build_simple_gp_model(drillholes)
    
    # Get cross-validation predictions
    cv_predictions = cross_val_predict(gp, X_normalized, y, cv=5)
    
    # Transform back to grade space
    actual_grades = np.exp(y)
    predicted_grades = np.exp(cv_predictions)
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Panel 1: Actual vs Predicted scatter
    ax1.scatter(actual_grades, predicted_grades, s=50,
                color='white', edgecolors='black', linewidths=1, alpha=0.7)
    
    # Perfect prediction line
    max_val = max(actual_grades.max(), predicted_grades.max())
    ax1.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='Perfect Prediction')
    
    # Calculate R²
    r2 = 1 - np.sum((actual_grades - predicted_grades) ** 2) / np.sum((actual_grades - actual_grades.mean()) ** 2)
    mae = np.mean(np.abs(actual_grades - predicted_grades))
    
    # Add statistics text
    stats_text = f'R² = {r2:.3f}\nMAE = {mae:.3f} ppm'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1))
    
    # Apply minimalist style
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_position(("outward", 5))
    ax1.spines["bottom"].set_position(("outward", 5))
    ax1.set_title('Cross-Validation: Actual vs Predicted Grades', 
                  fontsize=12, fontweight="bold", loc="left")
    ax1.set_xlabel('Actual Grade (ppm)', fontsize=10)
    ax1.set_ylabel('Predicted Grade (ppm)', fontsize=10)
    ax1.legend(loc='lower right', frameon=False, fontsize=9)
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.set_aspect('equal')
    
    # Panel 2: Residuals distribution
    residuals = actual_grades - predicted_grades
    
    bins = np.linspace(residuals.min(), residuals.max(), 25)
    counts, edges = np.histogram(residuals, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    
    ax2.bar(bin_centers, counts, width=np.diff(edges)[0] * 0.9,
            color='white', edgecolor='black', linewidth=1.5)
    
    # Add zero line
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Zero Residual')
    
    # Add statistics
    mean_residual = residuals.mean()
    std_residual = residuals.std()
    
    stats_text = f'Mean = {mean_residual:.3f} ppm\nStd Dev = {std_residual:.3f} ppm'
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1))
    
    # Apply minimalist style
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_position(("outward", 5))
    ax2.spines["bottom"].set_position(("outward", 5))
    ax2.set_title('Prediction Residuals Distribution', 
                  fontsize=12, fontweight="bold", loc="left")
    ax2.set_xlabel('Residual (Actual - Predicted, ppm)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.legend(loc='upper left', frameon=False, fontsize=9)
    
    # Save
    save_fig('08_ore_grade_accuracy.png')
    print("✓ Created: 08_ore_grade_accuracy.png")

def main():
    """Generate all visualizations."""
    set_tufte_defaults()
    print("=" * 60)
    print("ORE GRADE FORECASTING - VISUALIZATION GENERATION")
    print("=" * 60)
    print()
    
    # Set serif font globally
    plt.rcParams['font.family'] = 'serif'
    
    print("Creating visualizations...")
    create_main_visualization()
    create_accuracy_visualization()
    
    print()
    print("=" * 60)
    print("All visualizations created successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

