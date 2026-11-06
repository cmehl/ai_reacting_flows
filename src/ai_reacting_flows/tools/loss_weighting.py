"""
Weights of the training loss based on the increments of the species (learning.loss_weighting_p, loss_weighting_eps).

With output_omegas and log_transform_Y: 1, the target of species k at point i is the increment
omega_ik = ln Y_k(t+dt) - ln Y_k(t) [-]. The MSE loss weighs a relative error r on omega like r^2 omega^2: for the
same relative error, small increments (burnt gases) count much less than large ones (reaction zone). The weights

    w_ik = (omega_ik^2 + eps^2)^(-p) / mean_train(...)_k

give back weight to small increments: p = 0 is the usual loss, p = 1 a relative error above eps (absolute error
below), p = 1/2 close to the inverse increment weights 1/(|omega| + eps). eps [-] (in ln Y per step) bounds the
weights: the ratio of weights is at most (max |omega| / eps)^(2p). Weights are normalized to a mean of 1 per species
on the training points. Points clipped at the threshold (Y_k at the threshold at t and t+dt: omega = 0 from the
clipping, not from chemistry) and species excluded from the transform have the weight 1 of the usual loss.
See docs/theory.md (Loss weighting).
"""

import numpy as np

# Tolerance on the transformed mass fraction to detect a value clipped at the threshold [-]
CLIP_TOL = 1.0e-9


def increment_weights(x_species_t, omega, log_threshold, weighted_cols, p, eps, mean_ref=None):
    """Loss weights (n_points, n_species), and the normalization (n_species) used.

    x_species_t: ln Y_k(t) clipped (unscaled species part of the network input) (n_points, n_species)
    omega: ln Y_k(t+dt) - ln Y_k(t) (unscaled network output) (n_points, n_species)
    log_threshold: ln(threshold), weighted_cols: indices of the species with ln transform
    p [-], eps [-]: exponent and floor of the weights
    mean_ref: normalization of the training points (n_species), None to compute it (training points)
    """

    w = (omega**2 + eps**2) ** (-p)

    # Points clipped at the threshold at t and t+dt
    clipped = (x_species_t <= log_threshold + CLIP_TOL) & (x_species_t + omega <= log_threshold + CLIP_TOL)

    # Normalization: mean 1 per species on the points that are not clipped
    if mean_ref is None:
        w_not_clipped = np.where(clipped, np.nan, w)
        mean_ref = np.nanmean(w_not_clipped, axis=0)
        mean_ref[np.isnan(mean_ref)] = 1.0   # Species clipped on all points
    w = w / mean_ref

    # Usual loss for clipped points and species without ln transform
    w[clipped] = 1.0
    unweighted_cols = [k for k in range(w.shape[1]) if k not in weighted_cols]
    w[:, unweighted_cols] = 1.0

    return w, mean_ref


def effective_fraction(w):
    """Effective fraction of points of weights w [-]: (sum w)^2 / (n sum w^2), 1 for equal weights."""

    return float(np.sum(w)**2 / (w.size * np.sum(w**2)))
