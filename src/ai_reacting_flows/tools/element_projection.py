"""
Projection of mass fractions on the conservation of the elements, for numpy arrays and torch tensors.

Shared by inference (NNInference, yk_renormalization: 3) and training (NN_manager, learning.project_elements),
so that the networks are trained with the correction used in simulations. See docs/theory.md.
"""

import numpy as np
import torch


def project_elements(Y_old, Y_new, a, c):
    """Mass fractions Y_new [-] (n, n_species) corrected to keep the element mass fractions of Y_old: extension
    of Wan et al., Combust. Flame 220 (2020), Eqs. (7)-(8), https://doi.org/10.1016/j.combustflame.2020.06.008,
    to all elements, O included (the paper corrects O with O2 only, which fails in rich mixtures where O2 vanishes).

    a: a_Ai = n_Ai W_A / W_i [-] (n_elements, n_species), c: c_Ai = 1 if species i contains A, else 0.

    The correction of each species is proportional to its mass fraction, with one multiplier mu_A per
    element A it contains: dY_i = Y_i sum_A mu_A c_Ai. The element mass fractions are linear in Y,
    Y_A = sum_i a_Ai Y_i, so the multipliers are solved together (one system per state, exact in one step):
        sum_B M_AB mu_B = Y_A(t) - Y_A(t+dt),     M_AB = sum_i a_Ai Y_i c_Bi   (Y at t+dt)
    For a single element, mu_A = (Y_A(t) - Y_A(t+dt)) / Y_A(t+dt): Eq. (8) of the paper.
    """

    xp = torch if isinstance(Y_new, torch.Tensor) else np

    # Element imbalance (n, n_elements) and matrices of the systems (n, n_elements, n_elements)
    dY_a = Y_old @ a.T - Y_new @ a.T
    M = xp.einsum("ai,ni,bi->nab", a, Y_new, c)

    # Elements absent from a state (or from the mechanism), M_AA = Y_A = 0: multiplier set to 0
    absent = xp.einsum("naa->na", M) <= 0.0
    for i_el in range(a.shape[0]):
        M[absent[:, i_el], i_el, :] = 0.0
        M[absent[:, i_el], :, i_el] = 0.0
        M[absent[:, i_el], i_el, i_el] = 1.0
    dY_a[absent] = 0.0

    # Multipliers (n, n_elements) and corrected mass fractions
    mu = xp.linalg.solve(M, dY_a[:, :, None])[:, :, 0]

    return Y_new * (1.0 + mu @ c)
