"""
Transforms of mass fractions and features of the networks and of k-means clustering.

Shared by database processing (LearningDatabase), training diagnostics (NN_manager) and
inference (NNTesting), so that the three steps use the same definitions.

Transform of mass fractions Y [-], chosen with log_transform:
    0: none
    1: logarithm, log(max(Y, threshold))
    2: Box-Cox, (max(Y, 0)**lambda_bct - 1) / lambda_bct
       (Box & Cox, J. R. Stat. Soc. B 26:211, 1964, https://doi.org/10.1111/j.2517-6161.1964.tb00553.x)
Species listed in log_excluded_species are kept on a linear scale. The time step dt [s] always
has its own logarithm (no threshold), whatever the transform of the mass fractions.

Inverse functions accept numpy arrays and torch tensors (training diagnostics run on tensors).
"""

import numpy as np
import torch

# Floor of the time step before its logarithm [s] (only avoids log(0))
DT_EPS = 1.0e-300


#==============================================================================
# MASS FRACTIONS
#==============================================================================

def clip(Y, log_transform, threshold):
    """Mass fractions [-] clipped before the transform: to threshold for log (avoids log(0)),
    to 0 for Box-Cox (negative values would give NaN), unchanged without transform."""

    # Logarithm
    if log_transform == 1:
        return np.maximum(Y, threshold)

    # Box-Cox
    elif log_transform == 2:
        return np.maximum(Y, 0.0)

    return Y


def forward(Y, log_transform, lambda_bct):
    """Transform of (clipped) mass fractions [-]."""

    # Logarithm
    if log_transform == 1:
        return np.log(Y)

    # Box-Cox
    elif log_transform == 2:
        return (Y**lambda_bct - 1.0) / lambda_bct

    return Y


def inverse(Y_t, log_transform, lambda_bct):
    """Inverse transform: mass fractions [-] from transformed values (numpy array or torch tensor)."""

    # Logarithm
    if log_transform == 1:
        return torch.exp(Y_t) if isinstance(Y_t, torch.Tensor) else np.exp(Y_t)

    # Box-Cox
    elif log_transform == 2:
        return (Y_t * lambda_bct + 1.0) ** (1.0 / lambda_bct)

    return Y_t


def transformed_columns(species, log_excluded_species):
    """Indices of the species that are transformed (not in log_excluded_species)."""

    return [j for j, spec in enumerate(species) if spec not in log_excluded_species]


def transform_species(Y, species, log_transform, threshold, lambda_bct, log_excluded_species):
    """Clipped and transformed mass fractions, array (n_points, n_species).

    Y: mass fractions [-] (n_points, n_species), species: names of the columns of Y.
    Columns of log_excluded_species are unchanged.
    """

    Y_t = np.array(Y, dtype=float)
    cols = transformed_columns(species, log_excluded_species)

    Y_t[:, cols] = forward(clip(Y_t[:, cols], log_transform, threshold), log_transform, lambda_bct)

    return Y_t


def inverse_species(Y_t, species, log_transform, lambda_bct, log_excluded_species):
    """Mass fractions [-] from transformed mass fractions (n_points, n_species), numpy array or torch tensor.

    Columns of log_excluded_species are unchanged.
    """

    Y = Y_t.clone() if isinstance(Y_t, torch.Tensor) else np.array(Y_t, dtype=float)
    cols = transformed_columns(species, log_excluded_species)

    Y[:, cols] = inverse(Y_t[:, cols], log_transform, lambda_bct)

    return Y


#==============================================================================
# TIME STEP
#==============================================================================

def transform_dt(dt):
    """Time step feature: log(dt), dt [s] (no threshold), whatever the transform of the mass fractions:
    time steps span orders of magnitude."""

    return np.log(np.maximum(dt, DT_EPS))


#==============================================================================
# SCALING
#==============================================================================

def scaler_std(var):
    """Standard deviations sqrt(var) with zeros replaced by 1, as in sklearn StandardScaler,
    which was used to scale the training data (LearningDatabase.process_database)."""

    std = np.sqrt(var)
    std[std == 0.0] = 1.0

    return std


def scale(x, mean, std):
    """Standardized values (x - mean) / std."""

    return (x - mean) / std


def unscale(x_scaled, mean, std):
    """Values from standardized values: mean + std * x_scaled."""

    return mean + std * x_scaled


#==============================================================================
# NETWORK OUTPUT
#==============================================================================

def output_to_Yk(x_species_t, y_t, species, output_omegas, log_transform_X, log_transform_Y, lambda_bct, log_excluded_species):
    """Mass fractions at t+dt [-] (n_points, n_species) from the network input and output.

    Both x_species_t and y_t must be UNSCALED (standardization removed with unscale), not the
    scaled values seen by the network.

    x_species_t: species part of the network input, unscaled: transformed (clipped) mass fractions at t
    y_t: network output, unscaled, as built in LearningDatabase.process_database:
        - output_omegas, log_transform_Y > 0: f(Y(t+dt)) - f(Y(t)), with f the transform
        - output_omegas, log_transform_Y = 0: Y(t+dt) - Y(t), with Y(t) clipped as in the input
        - otherwise: f(Y(t+dt))
    """

    # Outputs are differences in transformed space
    if output_omegas and log_transform_Y > 0:
        return inverse_species(x_species_t + y_t, species, log_transform_Y, lambda_bct, log_excluded_species)

    # Outputs are differences of mass fractions
    elif output_omegas:
        return inverse_species(x_species_t, species, log_transform_X, lambda_bct, log_excluded_species) + y_t

    # Outputs are mass fractions at t+dt
    return inverse_species(y_t, species, log_transform_Y, lambda_bct, log_excluded_species)


def network_target(Y_old, Y_new, species, output_omegas, log_transform_X, log_transform_Y, threshold, lambda_bct, log_excluded_species):
    """Network output (unscaled) for a step from Y_old to Y_new [-] (n_points, n_species), as built in
    LearningDatabase.process_database: the inverse of output_to_Yk.

        - output_omegas, log_transform_Y > 0: f(Y(t+dt)) - f(Y(t)), with f the transform
        - output_omegas, log_transform_Y = 0: Y(t+dt) - Y(t), with Y(t) clipped as in the input
        - otherwise: f(Y(t+dt))
    """

    Y_new_t = transform_species(Y_new, species, log_transform_Y, threshold, lambda_bct, log_excluded_species)

    # Differences of mass fractions, Y(t) clipped as in the input
    if output_omegas and log_transform_X > 0 and log_transform_Y == 0:
        Y_old_clipped = np.array(Y_old, dtype=float)
        cols = transformed_columns(species, log_excluded_species)
        Y_old_clipped[:, cols] = clip(Y_old_clipped[:, cols], log_transform_X, threshold)
        return Y_new_t - Y_old_clipped

    # Differences in transformed space (or of mass fractions without transform)
    elif output_omegas:
        return Y_new_t - transform_species(Y_old, species, log_transform_X, threshold, lambda_bct, log_excluded_species)

    # Mass fractions at t+dt
    return Y_new_t


#==============================================================================
# K-MEANS FEATURES
#==============================================================================

def kmeans_features(T, Y, dt, species, clusterize_on, log_transform_X, threshold, lambda_bct, log_excluded_species):
    """Features of k-means clustering, array (n_points, n_features), before standardization.

    T: temperature [K] (n_points), Y: mass fractions [-] (n_points, n_species) of species (N2 removed
    without N chemistry), dt: time step [s] (n_points), only used if clusterize_on is "dt" or "all".
    Features: "phys": [T, transformed Y], "dt": [log(dt)], "all": [T, transformed Y, log(dt)].
    """

    # Clustering on the time step only
    if clusterize_on == "dt":
        return np.reshape(transform_dt(np.asarray(dt, dtype=float)), (-1, 1))

    features = np.column_stack([T, transform_species(Y, species, log_transform_X, threshold, lambda_bct, log_excluded_species)])

    # Clustering on physical state and time step
    if clusterize_on == "all":
        features = np.column_stack([features, transform_dt(np.asarray(dt, dtype=float))])

    return features
