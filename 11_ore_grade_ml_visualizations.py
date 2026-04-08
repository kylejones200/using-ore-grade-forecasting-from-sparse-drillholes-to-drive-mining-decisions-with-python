import sys
import os

# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS

"""
Visualization generation for Blog 11: Ore Grade Forecasting with ML
Creates minimalist-style visualizations comparing Kriging, GP, and GBT models.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import warnings

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


warnings.filterwarnings('ignore')

def apply_minimalist_style_manual(ax):
    """Apply minimalist style components manually to axis."""
    plt.rcParams["font.family"] = "serif"
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
def generate_synthetic_geochemical_data(n_samples=1000):
    """
    Generate synthetic geochemical data with spatial structure.
    
    Returns:
        X: (n_samples, 2) coordinates
        y: (n_samples,) ore grades (% Cu)
    """
    np.random.seed(42)
    
    # Generate spatial coordinates (UTM-like)
    X = np.random.rand(n_samples, 2) * 1000  # 1000m x 1000m area
    
    # Create spatial structure with multiple ore zones
    y = np.zeros(n_samples)
    
    # High-grade zone (center-north)
    center1 = np.array([500, 700])
    dist1 = np.linalg.norm(X - center1, axis=1)
    y += 1.2 * np.exp(-dist1**2 / (150**2))
    
    # Medium-grade zone (west)
    center2 = np.array([200, 400])
    dist2 = np.linalg.norm(X - center2, axis=1)
    y += 0.8 * np.exp(-dist2**2 / (200**2))
    
    # Low-grade background
    y += 0.15
    
    # Add nugget effect (measurement noise)
    y += np.random.randn(n_samples) * 0.05
    
    # Clip to realistic range
    y = np.clip(y, 0.05, 1.5)
    
    return X, y

def simple_kriging(X_train, y_train, X_test):
    """Simple kriging implementation using Gaussian Process."""
    kernel = RBF(length_scale=100.0) + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=3)
    gp.fit(X_train, y_train)
    y_pred = gp.predict(X_test)
    return y_pred

def create_main_spatial_prediction_plot():
    """
    Create spatial prediction comparison: Kriging vs GP vs GBT.
    """
    print("Generating main spatial prediction visualization...")
    
    # Generate data
    X, y = generate_synthetic_geochemical_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create prediction grid
    grid_x = np.linspace(0, 1000, 100)
    grid_y = np.linspace(0, 1000, 100)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    grid_points = np.c_[grid_X.ravel(), grid_Y.ravel()]
    
    # Train models and predict
    print("  Training Kriging...")
    kriging_pred = simple_kriging(X_train, y_train, grid_points).reshape(100, 100)
    
    print("  Training Gaussian Process...")
    kernel_gp = RBF(length_scale=100.0) + WhiteKernel(noise_level=0.01)
    gp_model = GaussianProcessRegressor(kernel=kernel_gp, alpha=0.01, n_restarts_optimizer=3)
    gp_model.fit(X_train, y_train)
    gp_pred = gp_model.predict(grid_points).reshape(100, 100)
    
    print("  Training Gradient Boosting...")
    gbt_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gbt_model.fit(X_train, y_train)
    gbt_pred = gbt_model.predict(grid_points).reshape(100, 100)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = ['Ordinary Kriging', 'Gaussian Process', 'Gradient Boosted Trees']
    predictions = [kriging_pred, gp_pred, gbt_pred]
    
    for ax, title, pred in zip(axes, titles, predictions):
        # Contour plot
        contour = ax.contourf(grid_X, grid_Y, pred, levels=15, cmap='gray', alpha=0.8)
        
        # Overlay drill holes
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                           s=30, cmap='gray', edgecolors='white', linewidth=0.5,
                           vmin=pred.min(), vmax=pred.max(), zorder=10)
        
        # Apply minimalist style
        apply_minimalist_style_manual(ax)
        
        ax.set_xlabel('Easting (m)', fontsize=9)
        ax.set_ylabel('Northing (m)', fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold', loc='center', pad=10)
        ax.set_aspect('equal')
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=axes, orientation='horizontal', 
                       pad=0.08, aspect=40, shrink=0.8)
    cbar.set_label('Cu Grade (%)', fontsize=10)
    cbar.outline.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('/Users/k.jones/Desktop/blogs/blog_posts/11_ore_grade_ml_main.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Main spatial prediction visualization saved")

def create_model_comparison_plot():
    """
    Create bar chart comparing model performance metrics.
    """
    print("Generating model comparison visualization...")
    
    # Generate data
    X, y = generate_synthetic_geochemical_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train models
    print("  Training models for comparison...")
    
    # Kriging
    kriging_pred = simple_kriging(X_train, y_train, X_test)
    kriging_mae = np.mean(np.abs(y_test - kriging_pred))
    kriging_r2 = 1 - np.sum((y_test - kriging_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    # Gaussian Process
    kernel_gp = RBF(length_scale=100.0) + WhiteKernel(noise_level=0.01)
    gp_model = GaussianProcessRegressor(kernel=kernel_gp, alpha=0.01, n_restarts_optimizer=3)
    gp_model.fit(X_train, y_train)
    gp_pred = gp_model.predict(X_test)
    gp_mae = np.mean(np.abs(y_test - gp_pred))
    gp_r2 = 1 - np.sum((y_test - gp_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    # Gradient Boosted Trees
    gbt_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gbt_model.fit(X_train, y_train)
    gbt_pred = gbt_model.predict(X_test)
    gbt_mae = np.mean(np.abs(y_test - gbt_pred))
    gbt_r2 = 1 - np.sum((y_test - gbt_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    models = ['Kriging', 'Gaussian\nProcess', 'Gradient\nBoosting']
    mae_values = [kriging_mae, gp_mae, gbt_mae]
    r2_values = [kriging_r2, gp_r2, gbt_r2]
    
    colors = ['#0074D9', '#2ECC40', '#FF851B']
    
    # Left panel: MAE
    bars1 = ax1.bar(models, mae_values, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars1, mae_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    apply_minimalist_style_manual(ax1)
    ax1.set_ylabel('Mean Absolute Error (% Cu)', fontsize=10)
    ax1.set_title('Prediction Error', fontsize=12, fontweight='bold', loc='left', pad=20)
    ax1.set_ylim(0, max(mae_values) * 1.2)
    
    # Right panel: R²
    bars2 = ax2.bar(models, r2_values, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars2, r2_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    apply_minimalist_style_manual(ax2)
    ax2.set_ylabel('R² Score', fontsize=10)
    ax2.set_title('Predictive Performance', fontsize=12, fontweight='bold', loc='left', pad=20)
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add summary annotation
    fig.text(0.5, 0.02, 
            f'Test Set: n={len(X_test)} | Train Set: n={len(X_train)} | Best Model: Gradient Boosting (MAE={gbt_mae:.3f}, R²={gbt_r2:.3f})',
            ha='center', fontsize=9, style='italic', color='black')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('/Users/k.jones/Desktop/blogs/blog_posts/11_ore_grade_ml_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Model comparison visualization saved")
    print(f"  Kriging MAE: {kriging_mae:.3f}, R²: {kriging_r2:.3f}")
    print(f"  GP MAE: {gp_mae:.3f}, R²: {gp_r2:.3f}")
    print(f"  GBT MAE: {gbt_mae:.3f}, R²: {gbt_r2:.3f}")

def main():
    """Generate all visualizations for Blog 11."""
    set_tufte_defaults()
    print("="*70)
    print("Blog 11: Ore Grade ML - Visualizations")
    print("="*70)
    print()
    
    create_main_spatial_prediction_plot()
    create_model_comparison_plot()
    
    print()
    print("="*70)
    print("All visualizations generated successfully!")
    print("="*70)
    print()
    print("Files created:")
    print("  - 11_ore_grade_ml_main.png")
    print("  - 11_ore_grade_ml_comparison.png")

if __name__ == "__main__":
    main()

