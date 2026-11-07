"""
Weights of the training loss based on the increments of the species (learning.loss_weighting_p, loss_weighting_eps,
loss_weighting_q, loss_weighting_min_count).

With output_omegas and log_transform_Y: 1, the target of species k at point i is the increment
omega_ik = ln Y_k(t+dt) - ln Y_k(t) [-], scaled by the standard deviation sigma_k of the cluster. If all species of a
point have the same relative error e_i, the MSE loss of the point is e_i^2 Omega_i^2, with the norm of the scaled increments

    Omega_i^2 = sum_k (omega_ik / sigma_k)^2

For the same relative error, points with small increments (burnt gases) count much less than points with large ones
(reaction zone), and bins of Omega with many points count more than rare ones. The weights, one per point (same for
all its species),

    w_i = (Omega_i^2 + eps^2)^(-p) * max(N_bin(i), min_count)^(-q) / mean_train(...)

correct both: p = 0 is the usual loss, p = 1 a relative error above eps (absolute error below); N_bin(i) is the number
of training points in the bin of log10(max(Omega_i, eps)) of point i, q = 1 gives each bin the same share of points,
as an undersampling on Omega (LearningDatabase.undersample_Omega) without discarding points. eps [-] is in units of
sigma_k; min_count bounds the weights of rare bins. Weights are normalized to a mean of 1 on the training points.
See docs/theory.md (Loss weighting).
"""

import numpy as np

import ai_reacting_flows.tools.utilities as utils


def increment_norm(omega, sigma):
    """Norm Omega [-] (n_points) of the increments omega (n_points, n_species) scaled by their standard deviations
    sigma (n_species): sqrt(sum_k (omega_k / sigma_k)^2). The mean of the increments is not removed: Omega = 0 means
    no change of the state."""

    return np.sqrt(np.sum((omega / sigma)**2, axis=1))


def increment_weights(Omega, p, eps, q=0.0, min_count=1, ref=None):
    """Loss weights (n_points), and the reference of the training points used (dict: bin edges and numbers of
    points of log10(Omega), mean of the weights before normalization).

    Omega: norm of the scaled increments of each point (increment_norm) (n_points)
    p [-], eps [-]: exponent and floor of the increment factor
    q [-], min_count: exponent of the density factor, and floor of the numbers of points per bin
    ref: reference of the training points, None to compute it (training points)
    """

    log_Omega = np.log10(np.maximum(Omega, eps))

    # Bins of log10(Omega) of the training points (Freedman-Diaconis width, as undersample_1D)
    if ref is None:
        binwidth = utils.smart_binwidth(log_Omega)
        edges = np.arange(log_Omega.min(), log_Omega.max() + binwidth, binwidth)
        counts, _ = np.histogram(log_Omega, bins=edges)
        ref = {"edges": edges, "counts": counts}

    # Number of training points in the bin of each point (points outside the bins: first or last bin)
    i_bin = np.clip(np.digitize(log_Omega, ref["edges"][1:-1]), 0, len(ref["counts"]) - 1)
    N_bin = np.maximum(ref["counts"][i_bin], min_count)

    w = (Omega**2 + eps**2) ** (-p) * N_bin.astype(float) ** (-q)

    # Normalization: mean 1 on the training points
    if "mean" not in ref:
        ref["mean"] = np.mean(w)

    return w / ref["mean"], ref


def effective_fraction(w):
    """Effective fraction of points of weights w [-]: (sum w)^2 / (n sum w^2), 1 for equal weights."""

    return float(np.sum(w)**2 / (w.size * np.sum(w**2)))
