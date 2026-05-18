"""Ore-grade / permeability geomodeling demo using Gaussian process regression."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def main() -> None:
    rng = np.random.default_rng(42)
    n = 200
    coords = rng.uniform(0, 1, size=(n, 3))
    grade = (
        100 * np.exp(-3 * coords[:, 0])
        + 50 * np.sin(2 * np.pi * coords[:, 1])
        + rng.normal(0, 5, size=n)
    )
    grade = np.clip(grade, 1.0, None)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X = x_scaler.fit_transform(coords)
    y = y_scaler.fit_transform(grade.reshape(-1, 1)).ravel()
    kernel = RBF(length_scale=[0.5, 0.5, 0.5]) + Matern(length_scale=0.5, nu=1.5)
    gpr = GaussianProcessRegressor(
        kernel=kernel, alpha=0.1, n_restarts_optimizer=3, random_state=42
    )
    gpr.fit(X, y)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        model = GaussianProcessRegressor(kernel=kernel, alpha=0.1, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        y_val = y_scaler.inverse_transform(y[val_idx].reshape(-1, 1)).ravel()
        pred_val = y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
        scores.append(r2_score(y_val, pred_val))

    print(f"Cross-validated R² (mean): {np.mean(scores):.3f}")
    pred = y_scaler.inverse_transform(gpr.predict(X).reshape(-1, 1)).ravel()
    rmse = np.sqrt(mean_squared_error(grade, pred))
    print(f"In-sample RMSE: {rmse:.2f}")
    plt.figure(figsize=(8, 4))
    plt.scatter(grade, pred, alpha=0.6)
    plt.xlabel("Observed grade")
    plt.ylabel("GPR prediction")
    plt.title("Sparse drillhole GPR interpolation")
    plt.tight_layout()
    plt.savefig("ore_grade_gpr_fit.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
