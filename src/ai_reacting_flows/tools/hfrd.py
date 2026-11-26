"""Data augmentation by random perturbation of the states of a database (HFRD method).

HFRD method of Ding et al., Combust. Flame 231, 2021, https://doi.org/10.1016/j.combustflame.2021.111493:
temperature [K] and mass fractions [-] of a state are perturbed, with one random number for the temperature and
one per species. The temperature and the mass fraction of N2 change by at most 1/8 of their range in the reference
states; the mass fractions of the other species are changed by a power law, Y_k^(1 + d/10) with d in [-1, 1], and
normalised. N2 treated as in Zhang et al., https://arxiv.org/abs/2312.05800. The mechanism must include N2.

A perturbed state is kept if its temperature is above 300 K and if its mass fractions of elements (C, H, O, N) [-]
are in the range of the reference states (with a relative margin).

Shared by the canonical flames (database_flmts.py, reference states: flamelet states of the same reactor type) and
the stochastic reactors (particles_cloud.py, reference states: all the states of the database).
"""

import numpy as np


# Perturbation of the temperature and of N2: at most 1/A of their range
A = 8

# Power law of the other species: Y_k^(1 + d/B)
B = 10

# Number of draws before giving up a state
MAX_TRIES = 11

# Minimum temperature [K] of a perturbed state
T_MIN = 300.0


def hfrd_bounds(T, Y, elements_matrix, i_N2, margin=0.01):
    """Ranges of the reference states used by perturb_state.

    T: temperatures [K] (n,), Y: mass fractions [-] (n, n_species), elements_matrix: (n_elements, n_species) (see
    utilities.get_elements_matrix), i_N2: index of N2. The relative margin keeps a non-zero range of the mass
    fractions of elements for a database with a single mixture composition.

    Returns a dict: T_range [K] and Y_N2_range [-] (max - min), Z_el_min and Z_el_max [-] (n_elements,).
    """

    Z_el = Y @ elements_matrix.T

    bounds = {
        "T_range": T.max() - T.min(),
        "Y_N2_range": Y[:, i_N2].max() - Y[:, i_N2].min(),
        "Z_el_min": Z_el.min(axis=0) * (1.0 - margin),
        "Z_el_max": Z_el.max(axis=0) * (1.0 + margin),
    }

    return bounds


def perturb_state(T, Y, bounds, elements_matrix, i_N2, rng):
    """Perturbed state (T_new [K], Y_new [-]) of the state (T, Y), or None if no draw is accepted in MAX_TRIES.

    bounds: from hfrd_bounds. rng: numpy random generator (one per state for results independent of the
    number of MPI processes).
    """

    Y = np.clip(Y, 0, 1)
    nb_species = len(Y)

    for _ in range(MAX_TRIES):

        # Random numbers: c for the temperature, d for the mass fractions (one per species)
        c = rng.uniform(-1, 1)
        d = rng.uniform(-1, 1, size=nb_species)

        # New temperature
        T_new = T + (c / A) * bounds["T_range"]

        # New mass fraction of N2: as the temperature, a power law would change this major species too much
        Y_N2_new = Y[i_N2] + (d[i_N2] / A) * bounds["Y_N2_range"]

        # New mass fractions of the other species, normalised so that the sum with N2 is 1
        Y_new = Y**(1.0 + d / B)
        Y_new[i_N2] = 0.0
        Y_new = Y_new * (1.0 - Y_N2_new) / Y_new.sum()
        Y_new[i_N2] = Y_N2_new

        # Rejected if too cold
        if T_new < T_MIN:
            continue

        # Rejected if the mass fraction of N2 is not in [0, 1]
        if Y_N2_new < 0.0 or Y_N2_new > 1.0:
            continue

        # Rejected if a mass fraction of element (C, H, O, N) is outside the range of the reference states
        Z_el_new = elements_matrix @ Y_new
        if np.any(Z_el_new < bounds["Z_el_min"]) or np.any(Z_el_new > bounds["Z_el_max"]):
            continue

        return T_new, Y_new

    return None
