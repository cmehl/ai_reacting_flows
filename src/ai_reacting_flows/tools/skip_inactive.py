"""Inactive states: states whose chemistry over dt is negligible (fresh gas, burnt gas at equilibrium).

A CFD code does not integrate the chemistry of these states (skip_inactive of OneDCombustion, criterion "log"):
they are removed from the learning database (LearningDatabase.skip_inactive_points) and kept unchanged by the ANN
in the tests (skip_inactive of nn_testing.yaml).

The change over dt [s] is estimated by an explicit Euler step with the net production rates w_k [kmol/m3/s] of the
initial state:
    dY_k = |w_k W_k| dt / rho,    dT = |sum_k h_k w_k| dt / (rho cp)
A state is active if dT > tol_T [K] or max_k dY_k / max(Y_k, Y_floor) > tol_rel. The second test is the relative
change |d ln Y_k| = dt / tau_k, with tau_k = rho Y_k / |w_k W_k| the chemical time of species k: it sees the
exponential growth of radicals from near zero (induction period of autoignition). See docs/theory.md.
"""

import numpy as np


def active_states(gas, T, Y, P, dt, tol_T=1.0e-4, tol_rel=1.0e-3, Y_floor=1.0e-20):
    """Boolean mask (n,) of the active states.

    gas: Cantera Solution (its state is changed); T (n,) [K]; Y (n, n_species) [-] in the species order of gas;
    P (n,) or scalar [Pa]; dt (n,) or scalar [s]. Thresholds tol_T [K], tol_rel [-], Y_floor [-] of OneDCombustion.
    """

    n = len(T)
    P = np.broadcast_to(P, (n,))
    dt = np.broadcast_to(dt, (n,))
    W = gas.molecular_weights   # [kg/kmol]

    active = np.ones(n, dtype=bool)
    for i in range(n):
        gas.TPY = T[i], P[i], Y[i]
        wdot = gas.net_production_rates   # [kmol/m3/s]

        # Euler estimates of the changes of Y_k [-] and T [K] over dt
        dY = np.abs(wdot * W) * dt[i] / gas.density
        dT = abs(np.dot(gas.partial_molar_enthalpies, wdot)) * dt[i] / (gas.density * gas.cp_mass)

        species_active = np.max(dY / np.maximum(Y[i], Y_floor)) > tol_rel
        active[i] = species_active or dT > tol_T

    return active
