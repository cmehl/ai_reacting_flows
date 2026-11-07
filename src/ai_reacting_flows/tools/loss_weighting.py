"""
Weights of the training loss based on the increments of the species (learning.loss_weighting_p, loss_weighting_eps).

With output_omegas and log_transform_Y: 1, the target of species k at point i is the increment
omega_ik = ln Y_k(t+dt) - ln Y_k(t) [-], scaled by the standard deviation sigma_k of the cluster. If all species of a
point have the same relative error e_i, the MSE loss of the point is e_i^2 Omega_i^2, with the norm of the scaled increments

    Omega_i^2 = sum_k (omega_ik / sigma_k)^2

For the same relative error, points with small increments (burnt gases) count much less than points with large ones
(reaction zone). The weights, one per point (same for all its species),

    w_i = (Omega_i^2 + eps^2)^(-p) / mean_train(...)

give back weight to small increments: p = 0 is the usual loss, p = 1 a relative error above eps (absolute error
below). eps [-] is in units of sigma_k and bounds the weights: their ratio is at most (max Omega / eps)^(2p).
Weights are normalized to a mean of 1 on the training points. See docs/theory.md (Loss weighting).
"""

import numpy as np


def increment_norm(omega, sigma):
    """Norm Omega [-] (n_points) of the increments omega (n_points, n_species) scaled by their standard deviations
    sigma (n_species): sqrt(sum_k (omega_k / sigma_k)^2). The mean of the increments is not removed: Omega = 0 means
    no change of the state."""

    return np.sqrt(np.sum((omega / sigma)**2, axis=1))


def increment_weights(Omega, p, eps, mean_ref=None):
    """Loss weights (n_points), and the normalization used.

    Omega: norm of the scaled increments of each point (increment_norm) (n_points)
    p [-], eps [-]: exponent and floor of the weights
    mean_ref: normalization of the training points, None to compute it (training points)
    """

    w = (Omega**2 + eps**2) ** (-p)

    # Normalization: mean 1 on the training points
    if mean_ref is None:
        mean_ref = np.mean(w)

    return w / mean_ref, mean_ref


def effective_fraction(w):
    """Effective fraction of points of weights w [-]: (sum w)^2 / (n sum w^2), 1 for equal weights."""

    return float(np.sum(w)**2 / (w.size * np.sum(w**2)))
