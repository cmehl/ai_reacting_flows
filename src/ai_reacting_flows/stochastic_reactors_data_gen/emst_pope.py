"""
EMST mixing model of Subramaniam & Pope, with intermittent mixing (age property of the particles).

S. Subramaniam & S.B. Pope, A mixing model for turbulent reactive flows based on Euclidean minimum
spanning trees, Combust. Flame 115 (1998) 487-514, https://doi.org/10.1016/S0010-2180(98)00023-6
Python translation of the Fortran 90 implementation (emst.f, emst_subs.f) of Z. Ren, S. Subramaniam &
S.B. Pope, Implementation of the EMST mixing model (2002), http://tcg.mae.cornell.edu/emst

Particles i = 1..n have compositions f_ij (j = 1..nc), weights w_i and an age property a_i:
    a_i > 0: the particle is mixing, for a remaining normalized time a_i,
    a_i < 0: the particle is not mixing, for a remaining normalized time |a_i|.
Times are normalized by the mixing frequency: s = Omega t (in ParticlesCloud, Omega = 1 / mixing_time).
The tree is built on the scaled compositions g_ij = f_ij / fscale_j of the mixing particles. Mixing along
its edges makes the variance function VF (trace of the covariance of g) decay as exp(-s), in expectation.
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
from scipy.spatial.distance import pdist, squareform

# Durations of the non-mixing (Z0) and mixing (Z1) states [-], uniform in [lower, upper] (emst.f)
Z0_LOWER, Z0_UPPER = 0.1666, 0.1667
Z1_LOWER, Z1_UPPER = 0.0178, 0.3157

# Model and control constants (emst_subs.f defaults)
GTMAX = 2.5             # Upper limit of the variance decay rate of the tree particles, relative to all particles
SDEV_SMALL = 1.0e-4     # IEM instead of EMST if the range of the scaled compositions is below 4 SDEV_SMALL
VAR_SMALL = 1.0e-12     # IEM instead of EMST if the largest variance of the scaled compositions is below
OMDT_MIN = 1.0e-6       # Below this normalized time, mixing over 2 OMDT_MIN accepted with a probability (rejection)
FPHTHR = 0.3            # Composition change of a particle in one EMST step limited to FPHTHR standard deviations,
RPHTHR = 1.0e-5         # standard deviations limited from below by RPHTHR times the largest one
MAX_SEARCH = 20         # Number of increases of eta to bracket the variance decay
NO_TREE_SCALE = 1.0e16  # fscale of a composition not used for the tree (emst.f convention)


#==============================================================================
# AGE PROPERTY
#==============================================================================

def sample_age(n, a, b, rng):
    """n samples of the remaining duration of a state whose durations are uniform in [a, b] (raninit):
    density proportional to 1 on [0, a), to (b - x) / (b - a) on [a, b] (stationary distribution)."""

    u = rng.random(n)
    p_a = 2.0 * a / (a + b)

    return np.where(u < p_a, u * (a + b) / 2.0, b - np.sqrt((b**2 - a**2) * (1.0 - u)))


def init_ages(n, rng):
    """Ages [-] (n) of n particles in the stationary state (emst_init): mixing with the probability
    p0 = <Z1> / (<Z0> + <Z1>), remaining time of the state drawn from its stationary distribution."""

    p0 = (Z1_LOWER + Z1_UPPER) / ((Z0_LOWER + Z0_UPPER) + (Z1_LOWER + Z1_UPPER))
    mixing = rng.random(n) <= p0

    ages = -sample_age(n, Z0_LOWER, Z0_UPPER, rng)
    ages[mixing] = sample_age(mixing.sum(), Z1_LOWER, Z1_UPPER, rng)

    return ages


def increment_ages(ages, ds, rng):
    """Ages [-] after a normalized time ds (statinc): remaining times decrease by ds; a particle whose
    state ends switches to the other state, with a new duration uniform in [Z1_LOWER, Z1_UPPER] (mixing)
    or [Z0_LOWER, Z0_UPPER] (non-mixing)."""

    u = rng.random(len(ages))
    ending = np.abs(ages) <= ds

    new_ages = ages - np.sign(ages) * ds

    # Non-mixing particles start mixing, mixing particles stop
    new_ages[ending & (ages < 0.0)] = Z1_LOWER + (Z1_UPPER - Z1_LOWER) * u[ending & (ages < 0.0)]
    new_ages[ending & (ages >= 0.0)] = -(Z0_LOWER + (Z0_UPPER - Z0_LOWER) * u[ending & (ages >= 0.0)])

    return new_ages


#==============================================================================
# TREE
#==============================================================================

def tree_edges(x):
    """Edges (n - 1, 2) of the Euclidean minimum spanning tree of the points x (n, nc), by Prim's
    algorithm from point 0 (emsti). Each edge is (parent, child), in the order the children are added,
    so that a parent always comes before its children."""

    n = len(x)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True

    # Squared distances between all points (n, n), computed once
    D2 = squareform(pdist(x, "sqeuclidean"))

    # Squared distance of each point to the tree (infinite once in the tree), and its nearest point in the tree
    dist2 = D2[0].copy()
    dist2[0] = np.inf
    nearest = np.zeros(n, dtype=int)

    edges = np.empty((n - 1, 2), dtype=int)
    for i_edge in range(n - 1):
        k = np.argmin(dist2)
        edges[i_edge] = nearest[k], k
        in_tree[k] = True
        dist2[k] = np.inf

        # Points closer to k than to the rest of the tree
        closer = ~in_tree & (D2[k] < dist2)
        dist2[closer] = D2[k][closer]
        nearest[closer] = k

    return edges


def edge_coefficients(edges, w):
    """Coefficients B (n - 1) of the edges of the tree of particles of weights w (n) (edgewt, bmod):
    B = 2 min(W, 1 - W), with W the fraction of the tree weight on the child side of the edge."""

    # Weight of the subtree of each node, accumulated from the leaves (children after parents in edges)
    subtree_w = w.copy()
    for parent, child in edges[::-1]:
        subtree_w[parent] += subtree_w[child]

    W = subtree_w[edges[:, 1]] / w.sum()

    return 2.0 * np.minimum(W, 1.0 - W)


def tree_laplacian(edges, B, n):
    """Laplacian (n, n) of the tree weighted by the edge coefficients B (sparse, csc):
    L_ii = sum of B of the edges of i, L_ij = -B of the edge (i, j)."""

    i, j = edges[:, 0], edges[:, 1]
    rows = np.concatenate([i, j, i, j])
    cols = np.concatenate([i, j, j, i])
    vals = np.concatenate([B, B, -B, -B])

    return sparse.csc_matrix((vals, (rows, cols)), shape=(n, n))


def implicit_mixing(phi, w, L, eta):
    """Compositions (n, nc) after mixing along the tree, by an implicit Euler step (implct) of
        w_i dphi_i/ds = -alpha sum_edges(i, j) B (phi_i - phi_j),     eta = alpha ds:
    (diag(w) + eta L) phi_new = diag(w) phi, with L the Laplacian of the tree (tree_laplacian)."""

    # diag(w) + eta L, with the sparsity of L (its diagonal is filled: every node has an edge)
    A = eta * L
    A.setdiag(A.diagonal() + w)

    return splu(A).solve(w[:, None] * phi)


#==============================================================================
# MIXING
#==============================================================================

def variance_function(g, w):
    """Variance function [-]: trace of the covariance matrix of the compositions g (n, nc), weights w (n)."""

    w = w / w.sum()
    mean = w @ g

    return np.sum(w @ (g - mean)**2)


def iem(g, w, ds):
    """IEM mixing over the normalized time ds: the variance function decays by exp(-ds)."""

    mean = (w @ g) / w.sum()

    return g - (1.0 - np.exp(-0.5 * ds)) * (g - mean)


def _tree_particles(phi, w, ages, rng):
    """Particles of the tree (pcheck): the mixing ones (ages > 0), plus non-mixing ones, closest to the end
    of their state first (they become mixing), until the variance function of the tree particles is large
    enough for the whole ensemble to decay at a rate at most GTMAX times theirs:
        p >= VF / (GTMAX VF_tree),     p the weight fraction of the tree particles.
    phi: fluctuations of the scaled compositions, w: normalized weights. ages modified in place."""

    vf = variance_function(phi, w)
    in_tree = ages > 0.0
    candidates = np.where(~in_tree)[0]
    candidates = candidates[np.argsort(np.abs(ages[candidates]))]

    for k in candidates:
        p = w[in_tree].sum()
        vf_tree = variance_function(phi[in_tree], w[in_tree]) if in_tree.any() else 0.0
        p_min = min(vf / (vf_tree * GTMAX), 1.0) if vf_tree > 0.0 else 1.0
        if p >= p_min:
            break

        # Particle k starts mixing
        ages[k] = sample_age(1, Z1_LOWER, Z1_UPPER, rng)[0]
        in_tree[k] = True

    return in_tree


def emst_step(g, w, ages, ds, rng):
    """One EMST mixing step (mixemst, rtfini) of the scaled compositions g (n, nc), normalized weights w,
    over at most the normalized time ds. Returns the compositions after mixing and the normalized time
    ds_done actually mixed (variance function decayed by exp(-ds_done)). ages modified in place."""

    # Threshold of the composition change of a particle
    sd = np.sqrt(w @ (g - w @ g)**2)
    dphi_max = FPHTHR * np.maximum(sd, RPHTHR * sd.max())

    # Fluctuations
    mean = w @ g
    phi = g - mean

    # Tree of the mixing particles, its edge coefficients and its Laplacian (the same for all values of eta)
    idx = np.where(_tree_particles(phi, w, ages, rng))[0]
    edges = tree_edges(phi[idx])
    B = edge_coefficients(edges, w[idx])
    L = tree_laplacian(edges, B, len(idx))

    # Mixing time of the step: at most the shortest duration of the mixing state
    ds_step = min(ds, Z1_LOWER)

    vf = variance_function(phi, w)
    vf_target = vf * np.exp(-ds_step)

    # Mixing of the tree particles with eta, the others unchanged; variance function in excess of the target
    def mix(eta):
        phi_new = phi.copy()
        phi_new[idx] = implicit_mixing(phi[idx], w[idx], L, eta)
        return phi_new, variance_function(phi_new, w) - vf_target

    # Estimate of eta from the initial decay rate of the variance function (rtest):
    # dVF/ds = -2 alpha sum_edges B |dphi|^2 = -VF
    dphi_edges = phi[idx][edges[:, 0]] - phi[idx][edges[:, 1]]
    alpha_est = 0.5 * vf / np.sum(B * np.sum(dphi_edges**2, axis=1))

    # Bracket the target: increase eta until the variance function is below the target (rtfini)
    eta_0 = 1.08 * alpha_est * ds_step
    eta = eta_0
    excess_0 = vf - vf_target
    phi_new, excess = mix(eta)
    for _ in range(MAX_SEARCH):
        if excess <= 0.0:
            break
        eta = 1.10 * eta if excess > 0.4 * excess_0 else 1.05 * eta / (1.0 - excess / excess_0)
        phi_new, excess = mix(eta)

    # Target not reached: rtfini restarts twice from 0.99 eta_0, which cannot reach it either (VF decreases
    # with eta), and goes on with the last of them (less mixing, accounted for by ds_done below)
    if excess > 0.0:
        eta = 0.99**2 * eta_0
        phi_new, excess = mix(eta)

    # Composition changes limited to dphi_max: eta halved until satisfied
    while np.any(np.max(np.abs(phi_new[idx] - phi[idx]), axis=0) > dphi_max):
        eta = eta / 2.0
        phi_new, excess = mix(eta)

    # Normalized time actually mixed; too much mixing accepted with the probability ds_step / ds_done
    ds_done = np.log(vf / (excess + vf_target))
    if ds_done > ds_step:
        if rng.random() >= ds_step / ds_done:
            phi_new = phi
        ds_done = ds_step

    return phi_new + mean, ds_done


def emst_mix(f, fscale, w, ages, ds, rng):
    """EMST mixing (emst, emst_scl) of the compositions f (n, nc) of particles of weights w (n) and ages
    (n), over the normalized time ds = Omega dt. The tree is built on g = f / fscale (a large fscale_j makes
    it independent of f_j). Returns the compositions after mixing and the incremented ages."""

    if len(w) < 2 or ds <= 0.0:
        return f, ages

    g = f / fscale
    w = w / w.sum()
    ages = ages.copy()

    # Largest variance and range of the scaled compositions
    def largest_variance_and_range(g):
        variances = w @ (g - w @ g)**2
        return variances.max(), np.max(g.max(axis=0) - g.min(axis=0))

    # Variance almost zero at the end of the step: IEM
    _, g_range = largest_variance_and_range(g)
    if g_range * np.exp(-ds) < SDEV_SMALL / 10.0:
        return iem(g, w, ds) * fscale, increment_ages(ages, ds, rng)

    ds_left = ds
    while ds_left > 0.0:

        # Small variance: IEM for the rest of the time
        g_var, g_range = largest_variance_and_range(g)
        if g_var < VAR_SMALL or g_range < 4.0 * SDEV_SMALL:
            g = iem(g, w, ds_left)
            ages = increment_ages(ages, ds_left, rng)
            break

        # Very small time left: mixing over 2 OMDT_MIN, accepted with a probability if too much
        if ds_left <= OMDT_MIN:
            g_try, ds_done = emst_step(g, w, ages, 2.0 * OMDT_MIN, rng)
            if ds_done >= ds_left:
                if rng.random() <= ds_left / ds_done:
                    g = g_try
                ages = increment_ages(ages, ds_left, rng)
                break
            g = g_try

        # Regular case
        else:
            g, ds_done = emst_step(g, w, ages, ds_left, rng)

        ages = increment_ages(ages, ds_done, rng)
        ds_left -= ds_done

    return g * fscale, ages
