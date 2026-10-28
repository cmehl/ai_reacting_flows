"""
Error metrics comparing the ANN to the reference (CVODE) in the tests of NNTesting (NN_testing.py).
"""

import numpy as np


def relative_error_percent(ref, ann):
    """Relative error |ann - ref| / ref [%], e.g. on the equilibrium temperature [K]."""

    return 100.0 * abs(ref - ann) / ref


def l2_error(x, omega_ann, omega_ref, x_window=None):
    """L2 norm of omega_ann - omega_ref over the grid x [m]: sqrt(sum((omega_ann - omega_ref)**2 dx)).

    Units of the error: units of omega times sqrt(m), e.g. reaction rates [1/s] of a 1D flame.
    x_window: optional (x_min, x_max) [m] restricting the norm, full grid if None.
    """

    # Spatial window selection
    if x_window is not None:
        x_min, x_max = x_window
        mask = (x >= x_min) & (x <= x_max)
    else:
        mask = slice(None)

    dx = np.gradient(x[mask])
    diff = omega_ann[mask] - omega_ref[mask]

    return float(np.sqrt(np.sum(diff**2 * dx)))


def mean_std(values):
    """Mean and standard deviation of a list of values, (0.0, 0.0) if empty."""

    arr = np.array(values)

    if arr.size == 0:
        return 0.0, 0.0

    return float(arr.mean()), float(arr.std())


def validation_errors(diff_scaled, std_Y):
    """MSE, MAE and RMSE per network output on the validation set.

    diff_scaled: prediction - target in scaled space (n_points, n_outputs), std_Y: standard
    deviations of the outputs scaler. "_scaled" metrics are in scaled space; "_phys" metrics are on
    the unscaled outputs (transformed space, see tools/transforms.py, not mass fractions): their
    difference is std_Y * diff_scaled, as the mean cancels.
    """

    diff_unscaled = diff_scaled * std_Y

    mse_scaled = np.mean(diff_scaled**2, axis=0)
    mse_phys = np.mean(diff_unscaled**2, axis=0)

    return {
        "mse_scaled": mse_scaled,
        "mae_scaled": np.mean(np.abs(diff_scaled), axis=0),
        "rmse_scaled": np.sqrt(mse_scaled),
        "mse_phys": mse_phys,
        "mae_phys": np.mean(np.abs(diff_unscaled), axis=0),
        "rmse_phys": np.sqrt(mse_phys),
    }
