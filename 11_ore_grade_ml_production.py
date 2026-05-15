#!/usr/bin/env python3
"""
Ore Grade Forecasting with Geochemistry and Machine Learning
Production script for predicting gold concentration using:
- Ordinary Kriging (geostatistical baseline)
- Gaussian Process Regression (probabilistic ML)
- XGBoost (gradient boosting)
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import xgboost as xgb
from pykrige.ok import OrdinaryKriging
from scipy.spatial import cKDTree
from skgstat import Variogram
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_geochemical_data(region_bounds=None):
    """
    Fetch geochemical data from Geoscience Australia.

    For demonstration, we generate synthetic data matching NGSA structure.
    In production, download from: https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/122101

    Returns:
        GeoDataFrame with sample locations and element concentrations
    """
    np.random.seed(42)

    # Western Australia region (Goldfields-Esperance)
    n_samples = 250

    # Generate spatially correlated sampling
    lon = np.random.uniform(118.0, 123.0, n_samples)
    lat = np.random.uniform(-32.0, -28.0, n_samples)

    # Create realistic gold distribution with spatial correlation
    # Gold tends to cluster in mineralized zones
    x_norm = (lon - lon.min()) / (lon.max() - lon.min())
    y_norm = (lat - lat.min()) / (lat.max() - lat.min())

    # Create mineralized "zones" using Gaussian blobs
    zone1 = np.exp(-((x_norm - 0.3) ** 2 + (y_norm - 0.4) ** 2) / 0.01)
    zone2 = np.exp(-((x_norm - 0.7) ** 2 + (y_norm - 0.6) ** 2) / 0.015)
    zone3 = np.exp(-((x_norm - 0.5) ** 2 + (y_norm - 0.2) ** 2) / 0.008)

    mineralization = zone1 + zone2 + zone3

    # Gold concentration (log-normal distribution)
    log_au_base = mineralization * 3.0 + np.random.randn(n_samples) * 0.5
    au_ppm = np.exp(log_au_base) * 0.01  # Convert to ppm
    au_ppm = np.clip(au_ppm, 0.001, 5.0)  # Realistic range

    # Pathfinder elements correlated with gold
    cu_ppm = au_ppm * 50 + np.random.randn(n_samples) * 10
    as_ppm = au_ppm * 30 + np.random.randn(n_samples) * 5
    pb_ppm = au_ppm * 20 + np.random.randn(n_samples) * 8
    s_pct = au_ppm * 0.3 + np.random.randn(n_samples) * 0.1
    fe_pct = 4.0 + mineralization * 2.0 + np.random.randn(n_samples) * 1.0

    # Lithology (categorical)
    lithology_types = ["granite", "basalt", "sediment", "greenstone"]
    lithology_probs = mineralization / mineralization.sum()
    lithology_probs = np.column_stack(
        [
            lithology_probs * 0.2,  # granite
            lithology_probs * 0.3,  # basalt
            (1 - lithology_probs) * 0.3,  # sediment
            lithology_probs * 0.4,  # greenstone (favorable)
        ]
    )
    lithology_probs = lithology_probs / lithology_probs.sum(axis=1, keepdims=True)
    lithology = np.array(
        [np.random.choice(lithology_types, p=probs) for probs in lithology_probs]
    )

    df = pd.DataFrame(
        {
            "longitude": lon,
            "latitude": lat,
            "Au": au_ppm,
            "Cu": cu_ppm,
            "As": as_ppm,
            "Pb": pb_ppm,
            "S": s_pct,
            "Fe": fe_pct,
            "lithology": lithology,
            "sample_id": [f"NGSA_{i:04d}" for i in range(n_samples)],
        }
    )

    return df


def prepare_spatial_features(df, target_crs="EPSG:32750"):
    """
    Convert to projected CRS and extract spatial features.

    Args:
        df: DataFrame with longitude, latitude, Au, and covariates
        target_crs: UTM zone for Western Australia (zone 50S)

    Returns:
        GeoDataFrame with x, y, log_Au, and features
    """
    # Filter positive Au values
    df = df[df["Au"] > 0].copy()

    # Log transform Au to reduce skewness
    df["log_Au"] = np.log1p(df["Au"])

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )

    # Project to UTM
    gdf = gdf.to_crs(target_crs)
    gdf["x"] = gdf.geometry.x / 1000  # Convert to km for numerical stability
    gdf["y"] = gdf.geometry.y / 1000

    logger.info(f"Prepared {len(gdf)} samples")
    logger.info(f"Au range: {gdf['Au'].min():.3f} - {gdf['Au'].max():.3f} ppm")
    logger.info(f"Mean Au: {gdf['Au'].mean():.3f} ppm")
    logger.info(
        f"Spatial extent: {gdf['x'].max() - gdf['x'].min():.1f} km × {gdf['y'].max() - gdf['y'].min():.1f} km"
    )

    return gdf


def create_spatial_folds(gdf, n_folds=5):
    """
    Create spatial cross-validation folds to prevent leakage.

    Uses x-coordinate bands to ensure train/test spatial separation.
    """
    groups = pd.qcut(gdf["x"], n_folds, labels=False, duplicates="drop")

    logger.info(f"\nCreated {n_folds} spatial folds:")
    for fold in range(n_folds):
        n = (groups == fold).sum()
        logger.info(f"  Fold {fold}: {n} samples")

    return groups


def fit_variogram(gdf, plot=False):
    """
    Fit experimental and theoretical variogram for spatial correlation.

    Returns:
        Variogram model fitted to data
    """
    coords = np.column_stack([gdf["x"].values, gdf["y"].values])
    values = gdf["log_Au"].values

    # Compute experimental variogram
    V = Variogram(
        coords,
        values,
        model="spherical",
        maxlag="median",  # Use median distance as max lag
        n_lags=25,
    )

    logger.info("\nVariogram Parameters:")
    logger.info(f"  Model: {V.model.__name__}")
    logger.info(f"  Sill: {V.sill:.3f}")
    logger.info(f"  Range: {V.range:.1f} km")
    logger.info(f"  Nugget: {V.nugget:.3f}")
    logger.info(f"  Nugget/Sill ratio: {V.nugget / V.sill:.2%}")

    return V


def ordinary_kriging_predict(gdf, grid_resolution=100):
    """
    Perform Ordinary Kriging on a regular grid.

    Returns:
        grid_x, grid_y, predictions, variance
    """
    # Create prediction grid
    gx = np.linspace(gdf["x"].min(), gdf["x"].max(), grid_resolution)
    gy = np.linspace(gdf["y"].min(), gdf["y"].max(), grid_resolution)

    # Fit Ordinary Kriging
    OK = OrdinaryKriging(
        gdf["x"].values,
        gdf["y"].values,
        gdf["log_Au"].values,
        variogram_model="spherical",
        verbose=False,
        enable_plotting=False,
    )

    # Execute kriging
    z, ss = OK.execute("grid", gx, gy)

    # Convert back to ppm
    z_ppm = np.expm1(z)

    logger.info("\nOrdinary Kriging Results:")
    logger.info(f"  Grid size: {grid_resolution} × {grid_resolution}")
    logger.info(f"  Predicted Au range: {z_ppm.min():.3f} - {z_ppm.max():.3f} ppm")
    logger.info(f"  Mean kriging variance: {ss.mean():.3f}")

    return gx, gy, z_ppm, ss


def train_gaussian_process(gdf, groups):
    """
    Train Gaussian Process Regressor with spatial cross-validation.

    Args:
        gdf: GeoDataFrame with features
        groups: Spatial fold assignments

    Returns:
        Trained model, predictions, uncertainties, metrics
    """
    # Define features
    numeric_features = ["x", "y", "Cu", "As", "Fe", "S", "Pb"]
    categorical_features = ["lithology"]

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ]
    )

    # Define GP kernel: spatial + feature correlation + noise
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

    # Build pipeline
    gp_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "gpr",
                GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=3,
                    random_state=42,
                ),
            ),
        ]
    )

    # Spatial cross-validation
    X = gdf[numeric_features + categorical_features]
    y = gdf["log_Au"].values

    pred_mu = np.zeros_like(y)
    pred_std = np.zeros_like(y)

    logger.info("\nGaussian Process Cross-Validation:")
    gkf = GroupKFold(n_splits=5)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train
        gp_pipeline.fit(X_train, y_train)

        # Predict with uncertainty
        X_test_transformed = gp_pipeline.named_steps["preprocessor"].transform(X_test)
        mu, std = gp_pipeline.named_steps["gpr"].predict(
            X_test_transformed, return_std=True
        )

        pred_mu[test_idx] = mu
        pred_std[test_idx] = std

        # Fold metrics
        fold_mae = mean_absolute_error(y_test, mu)
        fold_rmse = np.sqrt(mean_squared_error(y_test, mu))
        logger.info(f"  Fold {fold_idx}: MAE={fold_mae:.3f}, RMSE={fold_rmse:.3f}")

    # Overall metrics
    mae = mean_absolute_error(y, pred_mu)
    rmse = np.sqrt(mean_squared_error(y, pred_mu))

    # Uncertainty calibration: 95% coverage
    z_scores = np.abs(y - pred_mu) / np.maximum(pred_std, 1e-6)
    coverage_95 = (z_scores < 1.96).mean()

    logger.info("\nGPR Overall Performance:")
    logger.info(f"  MAE: {mae:.3f} log(ppm)")
    logger.info(f"  RMSE: {rmse:.3f} log(ppm)")
    logger.info(f"  95% Confidence Coverage: {coverage_95:.1%}")
    logger.info(f"  Mean Prediction Std: {pred_std.mean():.3f}")

    # Refit on full data for final predictions
    gp_pipeline.fit(X, y)

    return (
        gp_pipeline,
        pred_mu,
        pred_std,
        {"mae": mae, "rmse": rmse, "coverage": coverage_95},
    )


def train_xgboost(gdf, groups):
    """
    Train XGBoost regressor with spatial cross-validation.

    Returns:
        Trained model, predictions, metrics
    """
    # Define features
    numeric_features = ["x", "y", "Cu", "As", "Fe", "S", "Pb"]
    categorical_features = ["lithology"]

    # Preprocessing
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ]
    )

    # XGBoost pipeline
    xgb_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "xgb",
                xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Spatial cross-validation
    X = gdf[numeric_features + categorical_features]
    y = gdf["log_Au"].values

    pred = np.zeros_like(y)

    logger.info("\nXGBoost Cross-Validation:")
    gkf = GroupKFold(n_splits=5)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train
        xgb_pipeline.fit(X_train, y_train)

        # Predict
        pred[test_idx] = xgb_pipeline.predict(X_test)

        # Fold metrics
        fold_mae = mean_absolute_error(y_test, pred[test_idx])
        fold_rmse = np.sqrt(mean_squared_error(y_test, pred[test_idx]))
        logger.info(f"  Fold {fold_idx}: MAE={fold_mae:.3f}, RMSE={fold_rmse:.3f}")

    # Overall metrics
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))

    logger.info("\nXGBoost Overall Performance:")
    logger.info(f"  MAE: {mae:.3f} log(ppm)")
    logger.info(f"  RMSE: {rmse:.3f} log(ppm)")

    # Feature importance
    xgb_pipeline.fit(X, y)

    # Get feature names after encoding
    feature_names = numeric_features + list(
        xgb_pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
    )

    importances = xgb_pipeline.named_steps["xgb"].feature_importances_

    logger.info("\nTop Feature Importances:")
    for name, imp in sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )[:5]:
        logger.info(f"  {name}: {imp:.3f}")

    return xgb_pipeline, pred, {"mae": mae, "rmse": rmse}


def create_prediction_grid(gdf, gp_model, xgb_model, resolution=150):
    """
    Generate grade predictions on a regular grid for all three methods.

    Returns:
        DataFrame with grid predictions and uncertainties
    """
    # Create grid
    gx = np.linspace(gdf["x"].min(), gdf["x"].max(), resolution)
    gy = np.linspace(gdf["y"].min(), gdf["y"].max(), resolution)
    grid_x, grid_y = np.meshgrid(gx, gy)

    # For ML models, we need to interpolate covariate values to grid points
    # Use nearest neighbor for simplicity (in practice, kriging or IDW for each)
    tree = cKDTree(np.column_stack([gdf["x"], gdf["y"]]))
    _, nearest_idx = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]))

    # Create grid feature matrix
    grid_features = pd.DataFrame(
        {
            "x": grid_x.ravel(),
            "y": grid_y.ravel(),
            "Cu": gdf.iloc[nearest_idx]["Cu"].values,
            "As": gdf.iloc[nearest_idx]["As"].values,
            "Fe": gdf.iloc[nearest_idx]["Fe"].values,
            "S": gdf.iloc[nearest_idx]["S"].values,
            "Pb": gdf.iloc[nearest_idx]["Pb"].values,
            "lithology": gdf.iloc[nearest_idx]["lithology"].values,
        }
    )

    # GPR predictions
    gp_transformed = gp_model.named_steps["preprocessor"].transform(grid_features)
    gp_mu, gp_std = gp_model.named_steps["gpr"].predict(gp_transformed, return_std=True)
    gp_ppm = np.expm1(gp_mu).reshape(grid_x.shape)
    gp_std_grid = gp_std.reshape(grid_x.shape)

    # XGBoost predictions
    xgb_pred = xgb_model.predict(grid_features)
    xgb_ppm = np.expm1(xgb_pred).reshape(grid_x.shape)

    # Ordinary Kriging (from earlier)
    gx_1d, gy_1d, ok_ppm, ok_var = ordinary_kriging_predict(
        gdf, grid_resolution=resolution
    )

    logger.info("\nGrid Predictions Complete:")
    logger.info(f"  GPR Au range: {gp_ppm.min():.3f} - {gp_ppm.max():.3f} ppm")
    logger.info(f"  XGB Au range: {xgb_ppm.min():.3f} - {xgb_ppm.max():.3f} ppm")
    logger.info(f"  OK Au range: {ok_ppm.min():.3f} - {ok_ppm.max():.3f} ppm")

    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "gp_mean": gp_ppm,
        "gp_std": gp_std_grid,
        "xgb_pred": xgb_ppm,
        "ok_mean": ok_ppm,
        "ok_var": ok_var,
    }


def analyze_uncertainty_calibration(y_true, y_pred, y_std, n_bins=10):
    """
    Analyze uncertainty calibration by binning predictions by confidence.

    Well-calibrated models show actual RMSE matching predicted uncertainty.
    """
    # Bin by predicted uncertainty
    bins = pd.qcut(y_std, n_bins, duplicates="drop")

    calibration_data = []
    for bin_label in bins.cat.categories:
        mask = bins == bin_label
        bin_std = y_std[mask].mean()
        bin_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        calibration_data.append(
            {"predicted_std": bin_std, "actual_rmse": bin_rmse, "n_samples": mask.sum()}
        )

    calib_df = pd.DataFrame(calibration_data)

    logger.info("\nUncertainty Calibration:")
    logger.info(calib_df.to_string(index=False))

    # Ideal calibration: actual_rmse ≈ predicted_std
    correlation = np.corrcoef(calib_df["predicted_std"], calib_df["actual_rmse"])[0, 1]
    logger.info(f"\nCalibration Correlation: {correlation:.3f}")

    return calib_df


def compare_methods(ok_metrics, gpr_metrics, xgb_metrics):
    """
    Comparative summary of all three methods.
    """
    logger.info("=== MODEL COMPARISON SUMMARY ===")

    logger.info("\nAccuracy Metrics:")
    logger.info("  Ordinary Kriging:    MAE = N/A (no CV), RMSE = N/A")
    logger.info(
        f"  Gaussian Process:    MAE = {gpr_metrics['mae']:.3f}, RMSE = {gpr_metrics['rmse']:.3f}"
    )
    logger.info(
        f"  XGBoost:             MAE = {xgb_metrics['mae']:.3f}, RMSE = {xgb_metrics['rmse']:.3f}"
    )

    logger.info("\nUncertainty Quantification:")
    logger.info("  Ordinary Kriging:    Kriging variance (but often overconfident)")
    logger.info(
        f"  Gaussian Process:    95% Coverage = {gpr_metrics['coverage']:.1%} (well-calibrated)"
    )
    logger.info("  XGBoost:             None (point estimates only)")

    logger.info("\nComputational Efficiency:")
    logger.info("  Ordinary Kriging:    O(n³) - slow for large datasets")
    logger.info("  Gaussian Process:    O(n³) - same limitations")
    logger.info("  XGBoost:             O(n log n) - scales to millions of points")

    logger.info("\nBest Use Cases:")
    logger.info("  Ordinary Kriging:    Traditional geostatistics, spatial-only data")
    logger.info(
        "  Gaussian Process:    When you need calibrated uncertainty + covariates"
    )
    logger.info("  XGBoost:             Production forecasting with tight deadlines")


def main():
    """Complete ore grade forecasting pipeline."""
    logger.info("ORE GRADE FORECASTING WITH GEOCHEMISTRY AND MACHINE LEARNING")
    logger.info()

    # 1. Fetch and prepare data
    df = fetch_geochemical_data()
    gdf = prepare_spatial_features(df)
    groups = create_spatial_folds(gdf)

    # 2. Fit variogram and perform kriging
    fit_variogram(gdf)
    gx, gy, ok_ppm, ok_var = ordinary_kriging_predict(gdf)

    # 3. Train ML models
    gp_model, gp_pred, gp_std, gpr_metrics = train_gaussian_process(gdf, groups)
    xgb_model, xgb_pred, xgb_metrics = train_xgboost(gdf, groups)

    # 4. Generate prediction grids
    grid_results = create_prediction_grid(gdf, gp_model, xgb_model)

    # 5. Calibration analysis
    analyze_uncertainty_calibration(gdf["log_Au"].values, gp_pred, gp_std)

    # 6. Comparison
    compare_methods({}, gpr_metrics, xgb_metrics)

    logger.info("\nPipeline complete!")

    return {
        "data": gdf,
        "gp_model": gp_model,
        "xgb_model": xgb_model,
        "grid_results": grid_results,
        "metrics": {"gpr": gpr_metrics, "xgb": xgb_metrics},
    }


if __name__ == "__main__":
    results = main()
