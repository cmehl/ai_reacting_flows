"""
GENERATED USING CLAUDE SONNET 5
Hard constraint layer enforcing exact elemental-mass conservation while
guaranteeing strictly positive mass fractions, for the NN chemistry
surrogate models (MLPModel / DeepONet / DeepONet_shift).

--------------------------------------------------------------------------
IDEA
--------------------------------------------------------------------------
Given a (possibly signed, possibly unscaled) physical-space candidate
y_raw for the predicted mass fractions, we look for a correction of the
multiplicative ("exponential tilt") form:

    y_k = y0_k * exp( sum_e lambda_e * A[e,k] )

where y0 = softplus(y_raw) (or y_raw itself if it is already guaranteed
positive, e.g. it came out of a log/Box-Cox inverse transform), A is the
molar-mass atomic matrix (n_elements x n_species), and lambda is a small
per-batch-element vector of Lagrange multipliers (dimension = number of
elements tracked, typically 2-4: H, O, [N], [C]).

This form is positive for ANY lambda (it's a positive base times an
exponential), so positivity is guaranteed by construction -- we never
need to clip or renormalize afterwards.

lambda is chosen so that elemental mass is exactly conserved:

    A @ y = b_target      (b_target = atomic composition of the INPUT
                            state, which the true chemistry conserves
                            exactly)

This is a small nonlinear system (only as many unknowns as elements),
solved with a handful of Newton iterations, fully vectorized over the
batch and fully differentiable (so gradients flow back through the
correction into the network).

NOTE: since every species contributes its full mass to some element,
summing the per-element conservation equations over all elements also
gives you total mass conservation (sum_k y_k = 1) for free -- you do not
need a separate correction for that.
--------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConservationLayer(nn.Module):
    """
    Projects an unconstrained physical-space mass-fraction candidate onto
    the manifold of elementally-conserved, strictly positive compositions.

    Parameters
    ----------
    A_element : torch.Tensor, shape (n_elements, n_species)
        Molar-mass atomic matrix (same as NN_manager.A_element).
    n_newton_iters : int
        Number of Newton iterations used to solve for the Lagrange
        multipliers. 3-5 is typically enough given the well-scaled,
        low-dimensional (2-4) nature of the system.
    damping : float
        Small Tikhonov damping added to the Newton Jacobian for numerical
        stability (helps if some species have near-zero mass fraction,
        which can make the Jacobian ill-conditioned).
    eps : float
        Small floor added after softplus to keep y0 strictly > 0 (avoids
        log(0) / division issues if a species is predicted essentially
        absent).
    """

    def __init__(self, A_element: torch.Tensor, n_newton_iters: int = 5,
                 damping: float = 1e-6, eps: float = 1e-12):
        super().__init__()
        self.register_buffer("A_element", A_element)
        self.n_newton_iters = n_newton_iters
        self.damping = damping
        self.eps = eps

    def forward(self, y_raw: torch.Tensor, b_target: torch.Tensor,
                already_positive: bool = False):
        """
        y_raw : (batch, n_species)
            Unconstrained physical-space candidate mass fractions. Can be
            negative (e.g. omega mode: yval_in + predicted_delta, or a
            raw non-log-transformed output).
        b_target : (batch, n_elements)
            Target atomic mass per element, computed from the INPUT state:
            b_target = (A_element @ yval_in.T).T
            (chemistry exactly conserves this, so it is the correct
            right-hand side regardless of what the network predicts).
        already_positive : bool
            Set True when y_raw is already guaranteed > 0 by construction
            (i.e. it came from exp(...) or a well-behaved Box-Cox inverse
            of a log-transformed target, AND output_omegas is False).
            Set False whenever y_raw can be negative: this includes ALL
            omega-mode cases (output_omegas=True), regardless of whether
            log_transform_Y was applied to the omega itself, since the
            *sum* yval_in + predicted_delta is not guaranteed positive.

        Returns
        -------
        y_corrected : (batch, n_species)
            Strictly positive, exactly elementally-conserved prediction.
        lam : (batch, n_elements)
            Converged Lagrange multipliers (useful as an optional small
            regularization term: penalizing ||lam||^2 encourages the
            network to need only a small correction).
        """
        A = self.A_element  # (n_elements, n_species)

        y0 = y_raw if already_positive else (F.softplus(y_raw) + self.eps)

        batch_size, n_elements = b_target.shape
        lam = torch.zeros(batch_size, n_elements, dtype=y0.dtype, device=y0.device)
        eye = torch.eye(n_elements, dtype=y0.dtype, device=y0.device)

        for _ in range(self.n_newton_iters):
            tilt = torch.exp(torch.matmul(lam, A))            # (batch, n_species)
            y = y0 * tilt

            Ay = torch.matmul(y, A.T)                         # (batch, n_elements)
            residual = Ay - b_target

            # Jacobian: J[b,e,f] = sum_k A[e,k] * y[b,k] * A[f,k]
            weighted_A = A.unsqueeze(0) * y.unsqueeze(1)      # (batch, n_elements, n_species)
            J = torch.matmul(weighted_A, A.T)                 # (batch, n_elements, n_elements)
            J = J + self.damping * eye

            delta_lam = torch.linalg.solve(J, residual.unsqueeze(-1)).squeeze(-1)
            lam = lam - delta_lam

        tilt = torch.exp(torch.matmul(lam, A))
        y_corrected = y0 * tilt

        return y_corrected, lam


def compute_b_target(A_element: torch.Tensor, y_in: torch.Tensor) -> torch.Tensor:
    """
    Convenience helper: atomic composition target from the input state.

    A_element : (n_elements, n_species)
    y_in      : (batch, n_species)   physical-space input mass fractions
    returns   : (batch, n_elements)
    """
    return torch.matmul(y_in, A_element.T)


def forward_scale(y_physical: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                   log_transform: int, lambda_bct: float = 0.1) -> torch.Tensor:
    """
    Forward transform: physical space -> (log / Box-Cox transformed) -> standardized.

    This is the exact inverse of NN_manager._inverse_scale. Needed to bring a
    corrected, physical-space prediction back into scaled space so the loss
    can still be computed in the transformed/scaled space (as opposed to
    physical space).

    y_physical : tensor in physical space (must be > 0 if log_transform != 0)
    mean, std  : scaling statistics of the TRANSFORMED quantity (same stats
                 used by _inverse_scale)
    log_transform : 0 = none, 1 = plain log, 2 = Box-Cox-type transform
                    using lambda_bct. Bool is accepted for backward
                    compatibility (True -> 1).
    lambda_bct : Box-Cox exponent, only used when log_transform == 2.
    """
    if isinstance(log_transform, bool):
        log_transform = 1 if log_transform else 0

    if log_transform == 1:
        y_t = torch.log(y_physical)
    elif log_transform == 2:
        y_t = (torch.pow(y_physical, lambda_bct) - 1.0) / lambda_bct
    else:
        y_t = y_physical

    return (y_t - mean) / (std + 1e-7)


def corrected_prediction_to_scaled_target(y_corrected: torch.Tensor, yin_batch: torch.Tensor,
                                           mean: torch.Tensor, std: torch.Tensor,
                                           log_transform: int, output_omegas: bool,
                                           lambda_bct: float = 0.1) -> torch.Tensor:
    """
    Bring the conservation/positivity-corrected physical-space prediction
    back into scaled space, in the same representation as Y_train/Y_val
    (i.e. absolute mass fraction if output_omegas is False, delta/omega if
    output_omegas is True), so it can be compared against the scaled target
    with the usual loss_fn.

    y_corrected : (batch, n_species) physical-space, positive, conserved
                  prediction (output of ConservationLayer)
    yin_batch   : (batch, n_species) physical-space input mass fractions
    output_omegas : if True, the training target is the delta
                  (y_corrected - yin_batch) rather than the absolute value.

    NOTE: if output_omegas is True, log_transform must be 0 -- omegas can be
    negative and are therefore not log/Box-Cox transformable. This is
    asserted here defensively.
    """
    if output_omegas:
        if log_transform not in (0, False):
            raise ValueError(
                "log_transform must be 0 when output_omegas is True: omegas "
                "(deltas) can be negative and are not log/Box-Cox transformable."
            )
        quantity = y_corrected - yin_batch
    else:
        quantity = y_corrected

    return forward_scale(quantity, mean, std, log_transform, lambda_bct)