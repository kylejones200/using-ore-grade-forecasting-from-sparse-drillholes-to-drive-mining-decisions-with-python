#!/usr/bin/env python3
"""
Validation script for Blog 11: Ore Grade Forecasting with ML
Tests all functions and verifies outputs.
"""

import sys
import importlib.util

# Import module with numeric prefix
spec = importlib.util.spec_from_file_location(
    "ore_grade_ml_production",
    "/Users/k.jones/Desktop/blogs/blog_posts/11_ore_grade_ml_production.py"
)
production_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production_module)

# Import all functions
fetch_geochemical_data = production_module.fetch_geochemical_data
prepare_spatial_features = production_module.prepare_spatial_features
create_spatial_folds = production_module.create_spatial_folds
fit_variogram = production_module.fit_variogram
ordinary_kriging_predict = production_module.ordinary_kriging_predict
train_gaussian_process = production_module.train_gaussian_process
train_xgboost = production_module.train_xgboost
create_prediction_grid = production_module.create_prediction_grid
analyze_uncertainty_calibration = production_module.analyze_uncertainty_calibration
compare_methods = production_module.compare_methods

import numpy as np

# Import Tufte plotting utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tda_utils import setup_tufte_plot, TufteColors


def main():
    """Run validation tests for all functions."""
    print("="*70)
    print("BLOG 11 VALIDATION - ORE GRADE FORECASTING WITH ML")
    print("="*70)
    print()
    
    try:
        # Test 1: Data fetching
        print("TEST 1: Fetching geochemical data...")
        df = fetch_geochemical_data()
        assert len(df) == 250, "Expected 250 samples"
        assert 'Au' in df.columns, "Missing Au column"
        assert 'lithology' in df.columns, "Missing lithology column"
        print("✓ Data fetching successful\n")
        
        # Test 2: Spatial feature preparation
        print("TEST 2: Preparing spatial features...")
        gdf = prepare_spatial_features(df)
        assert 'x' in gdf.columns, "Missing x coordinate"
        assert 'y' in gdf.columns, "Missing y coordinate"
        assert 'log_Au' in gdf.columns, "Missing log_Au"
        print("✓ Spatial features prepared\n")
        
        # Test 3: Spatial folds
        print("TEST 3: Creating spatial cross-validation folds...")
        groups = create_spatial_folds(gdf)
        assert len(groups) == len(gdf), "Groups length mismatch"
        assert len(np.unique(groups)) >= 4, "Expected at least 4 folds"
        print("✓ Spatial folds created\n")
        
        # Test 4: Variogram fitting
        print("TEST 4: Fitting variogram...")
        V = fit_variogram(gdf)
        assert V.sill > 0, "Sill should be positive"
        assert V.range > 0, "Range should be positive"
        print("✓ Variogram fitted\n")
        
        # Test 5: Ordinary Kriging
        print("TEST 5: Performing Ordinary Kriging...")
        gx, gy, ok_ppm, ok_var = ordinary_kriging_predict(gdf, grid_resolution=50)
        assert ok_ppm.shape == (50, 50), "Unexpected grid shape"
        assert ok_ppm.min() >= 0, "Negative predictions found"
        print("✓ Ordinary Kriging completed\n")
        
        # Test 6: Gaussian Process
        print("TEST 6: Training Gaussian Process Regressor...")
        gp_model, gp_pred, gp_std, gpr_metrics = train_gaussian_process(gdf, groups)
        assert len(gp_pred) == len(gdf), "Prediction length mismatch"
        assert len(gp_std) == len(gdf), "Std length mismatch"
        assert gpr_metrics['mae'] > 0, "Invalid MAE"
        assert 0.8 <= gpr_metrics['coverage'] <= 1.0, "Coverage out of range"
        print("✓ Gaussian Process trained\n")
        
        # Test 7: XGBoost
        print("TEST 7: Training XGBoost...")
        xgb_model, xgb_pred, xgb_metrics = train_xgboost(gdf, groups)
        assert len(xgb_pred) == len(gdf), "Prediction length mismatch"
        assert xgb_metrics['mae'] > 0, "Invalid MAE"
        print("✓ XGBoost trained\n")
        
        # Test 8: Grid predictions
        print("TEST 8: Creating prediction grid...")
        grid_results = create_prediction_grid(gdf, gp_model, xgb_model, resolution=50)
        assert 'gp_mean' in grid_results, "Missing GPR mean"
        assert 'gp_std' in grid_results, "Missing GPR std"
        assert 'xgb_pred' in grid_results, "Missing XGB predictions"
        assert grid_results['gp_mean'].shape == (50, 50), "Grid shape mismatch"
        print("✓ Prediction grid created\n")
        
        # Test 9: Calibration analysis
        print("TEST 9: Analyzing uncertainty calibration...")
        calib_df = analyze_uncertainty_calibration(
            gdf["log_Au"].values, gp_pred, gp_std, n_bins=5
        )
        assert len(calib_df) >= 4, "Expected at least 4 calibration bins"
        assert 'predicted_std' in calib_df.columns, "Missing predicted_std"
        assert 'actual_rmse' in calib_df.columns, "Missing actual_rmse"
        print("✓ Calibration analysis completed\n")
        
        # Test 10: Method comparison
        print("TEST 10: Comparing methods...")
        compare_methods({}, gpr_metrics, xgb_metrics)
        print("✓ Method comparison completed\n")
        
        print("="*70)
        print("ALL VALIDATION TESTS PASSED!")
        print("="*70)
        print()
        
        # Summary statistics
        print("VALIDATION SUMMARY:")
        print(f"  Total samples: {len(gdf)}")
        print(f"  Spatial extent: {gdf['x'].max() - gdf['x'].min():.1f} × {gdf['y'].max() - gdf['y'].min():.1f} km")
        print(f"  Au range: {gdf['Au'].min():.3f} - {gdf['Au'].max():.3f} ppm")
        print(f"  GPR MAE: {gpr_metrics['mae']:.3f}")
        print(f"  GPR RMSE: {gpr_metrics['rmse']:.3f}")
        print(f"  GPR Coverage: {gpr_metrics['coverage']:.1%}")
        print(f"  XGB MAE: {xgb_metrics['mae']:.3f}")
        print(f"  XGB RMSE: {xgb_metrics['rmse']:.3f}")
        print(f"  XGB Improvement: {(1 - xgb_metrics['mae']/gpr_metrics['mae'])*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

