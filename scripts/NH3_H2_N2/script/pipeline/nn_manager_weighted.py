"""NN_manager variant with an inverse-increment weighted MSE (TFM_AMR burnt-gas bias study) and an optional element-conservation penalty.

networks_params.yaml, "learning" section:
    loss_weighting: inv_increment   # default "none" -> plain NNManagerReg / NN_manager behaviour
    element_penalty: 0.1            # default 0 -> no penalty (identical to the weighted loss alone)
    element_drift_ref: 1.0e-5       # drift scale of the penalty (CONVERGE hybrid_ann_tol)
    eps_file: eps_global.npy        # optional: global eps_k (e.g. median over the whole base) instead of the per-cluster median
loss_weighting and element_penalty also accept a list (one value per cluster), e.g. for the 3-zone T split
[none, inv_increment, inv_increment] = MSE at the cold edge of the front, weighted loss in the reaction zone and burnt gas.

The network target is the standardised log-increment omega_s = (omega - mean) / (std + 1e-7), omega = log Y(t+dt) - log Y(t)
(rollout: increment from the previous TRUE step, same scaler). Near equilibrium omega ~ 0, so a small constant offset of the
network costs almost nothing in a plain MSE while being 10-30 % of the true increment (a-priori diagnosis on post000107).
Weighted loss, per sample i and species k:
    w_ik = 1 / (|omega_ik| + eps_k),   eps_k = median_i |omega_ik| (training set)
    w_ik /= mean_i w_ik                 (per-species mean weight = 1 -> species balance of the loss unchanged)
    loss = mean_ik  w_ik * (pred_ik - target_ik)^2
omega is recovered from the target itself, so the same loss works for single-step and rollout batches.

Element penalty (2026-10-08): the weighted-loss network has a systematic element-conservation offset of ~1.3e-5 at 800-1100 K;
CONVERGE's renorm 3 then corrects it by spreading dY ~ Y over the species, which wipes out 10-12 % of the small chemical increment
per step at the cold edge of the front (or the hybrid sends those cells to CVODE). Penalty on the RAW output:
    d_e = (Z_e(Y_out) - Z_e(Y_in)) / Z_e(Y_in),  Z = A_element Y  (elements present in the mechanism only)
    loss += element_penalty * mean_i sum_e (d_e / element_drift_ref)^2
Y_in / Y_out are rebuilt from the scaled input and prediction exactly as in NN_manager's validation block (_inverse_scale,
_reconstruct_yk_abs). NN_manager calls loss_fn(pred, target) only, so the input batch is captured by a forward pre-hook on the model.
Implemented as a subclass (NN_manager untouched): train_model swaps the loss_fn it is given.
"""
import os

import numpy as np
import torch
import torch.nn as nn

from ai_reacting_flows.ann_model_generation.NN_manager_reg import NNManagerReg

N_STATS = 5_000_000  # rows used to estimate eps_k and the normalisation
N_DIAG = 1_000_000   # validation rows used for the element-drift diagnostics
LOG_EVERY = 10       # epochs between penalty log lines


EPS_MIN = 1e-12  # eps_k floor: a median |omega| of exactly 0 (species stuck at the log floor) would give 0/0


def raw_weight(omega, eps, mode, c):
    """inv_increment: 1/(|omega|+eps)  (unbounded contrast: large-|omega| cells can get ~0 weight)
    inv_increment_floor: 1 + c*eps/(|omega|+eps)  (in [1, 1+c]: small increments boosted, no cell below 1)"""
    if mode == "inv_increment":
        return 1.0 / (abs(omega) + eps)
    return 1.0 + c * eps / (abs(omega) + eps)


class InvIncrementWeightedMSE(nn.Module):

    def __init__(self, mean, std, eps, norm, mode="inv_increment", c=2.0):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("eps", eps)
        self.register_buffer("norm", norm)
        self.mode, self.c = mode, float(c)

    def forward(self, pred, target):
        omega = self.mean + (self.std + 1e-7) * target
        w = raw_weight(omega, self.eps, self.mode, self.c) / self.norm
        return torch.mean(w * (pred - target) ** 2)


class InputHolder:
    """Filled by a forward pre-hook with the last input batch given to the model."""
    x = None


class ElementPenaltyLoss(nn.Module):
    """base_loss(pred, target) + lam * mean_i sum_e (d_e / ref)^2, d_e = relative element drift of the RAW network output."""

    def __init__(self, base_loss, manager, holder, A, Xm, Xs, Ym, Ys, lam, ref):
        super().__init__()
        self.base, self.mgr, self.holder = base_loss, manager, holder
        self.register_buffer("A", A)
        self.register_buffer("Xm", Xm)
        self.register_buffer("Xs", Xs)
        self.register_buffer("Ym", Ym)
        self.register_buffer("Ys", Ys)
        self.lam, self.ref = float(lam), float(ref)
        self.acc = [0.0, 0.0, 0]  # training-batch sums of base loss and penalty, count

    def drift(self, X, pred):
        Xsp = X[:, 1:]
        yin = self.mgr._inverse_scale(Xsp, self.Xm[1:], self.Xs[1:], self.mgr.log_transform_X, cols=self.mgr.X_cols_all[1:])
        yout = self.mgr._reconstruct_yk_abs(Xsp, pred, self.Xm[1:], self.Xs[1:], self.Ym, self.Ys)
        zin = yin @ self.A.T
        zout = yout @ self.A.T
        return (zout - zin) / zin

    def forward(self, pred, target):
        base = self.base(pred, target)
        X = self.holder.x
        if X is None or X.shape[0] != pred.shape[0]:
            return base
        pen = torch.mean(torch.sum((self.drift(X, pred) / self.ref) ** 2, dim=1))
        if torch.is_grad_enabled():
            self.acc[0] += float(base.detach())
            self.acc[1] += float(pen.detach())
            self.acc[2] += 1
        return base + self.lam * pen


class NNManagerWeighted(NNManagerReg):

    def __init__(self, run_folder: str | None = None):
        super().__init__(run_folder)
        import oyaml as yaml
        import os
        with open(os.path.join(self.run_folder, "networks_params.yaml"), "r") as f:
            learning = yaml.safe_load(f)["learning"]
        # loss_weighting / element_penalty: one value for all clusters, or a list with one value per cluster
        self.loss_weighting_cfg = learning.get("loss_weighting", "none")
        self.loss_weight_c = float(learning.get("loss_weight_c", 2.0))
        self.element_penalty_cfg = learning.get("element_penalty", 0.0)
        self.element_drift_ref = float(learning.get("element_drift_ref", 1e-5))
        eps_file = learning.get("eps_file")          # optional: global eps_k (npy) instead of the per-cluster median
        if eps_file is not None and not os.path.isabs(eps_file):
            eps_file = os.path.join(self.run_folder, eps_file)
        self.eps_file = eps_file
        lw = self.loss_weighting_cfg if isinstance(self.loss_weighting_cfg, list) else [self.loss_weighting_cfg]
        for v in lw:
            if v not in ("none", "inv_increment", "inv_increment_floor"):
                raise ValueError(f"unknown loss_weighting {v}")
        self._log(f"WEIGHTED loss_weighting={self.loss_weighting_cfg}, loss_weight_c={self.loss_weight_c}, "
                  f"element_penalty={self.element_penalty_cfg}, element_drift_ref={self.element_drift_ref}, eps_file={self.eps_file}")

    @staticmethod
    def _per_cluster(cfg, i_cluster):
        return cfg[i_cluster] if isinstance(cfg, list) else cfg

    def _drift_diag(self, tag, model, pen_loss, X_val):
        n = min(N_DIAG, X_val.shape[0])
        X = torch.as_tensor(np.asarray(X_val[:n]), dtype=torch.float64, device=self.device)
        with torch.no_grad():
            d = torch.cat([pen_loss.drift(X[i:i + 200_000], model(X[i:i + 200_000])) for i in range(0, n, 200_000)])
        m = torch.max(torch.abs(d), dim=1).values
        self._log(f"ELEMENT_DRIFT {tag}: median max_e|d|={float(torch.median(m)):.3e}, "
                  f">1e-5: {100 * float((m > 1e-5).double().mean()):.3f} %, >2e-5: {100 * float((m > 2e-5).double().mean()):.3f} %, "
                  f"mean signed d per element=" + ", ".join(f"{float(v):+.2e}" for v in d.mean(dim=0)))

    def train_model(self, i_cluster, model, loss_fn, optimizer, scheduler, X_train, X_val, Y_train, Y_val,
                    Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std, Ymulti_train=None, Tmulti_train=None):
        t = lambda a: torch.tensor(np.asarray(a, dtype=np.float64), device=self.device)
        self.loss_weighting = self._per_cluster(self.loss_weighting_cfg, i_cluster)
        self.element_penalty = float(self._per_cluster(self.element_penalty_cfg, i_cluster))
        self._log(f"WEIGHTED cluster={i_cluster}: loss_weighting={self.loss_weighting}, element_penalty={self.element_penalty}")
        if self.loss_weighting in ("inv_increment", "inv_increment_floor"):
            mode, c = self.loss_weighting, self.loss_weight_c
            rng = np.random.default_rng(0)
            idx = rng.choice(Y_train.shape[0], min(N_STATS, Y_train.shape[0]), replace=False)
            Ys_np = Y_train[idx].detach().cpu().numpy() if torch.is_tensor(Y_train) else np.asarray(Y_train)[idx]
            omega = np.asarray(Yscaler_mean) + (np.asarray(Yscaler_std) + 1e-7) * Ys_np
            if self.eps_file is not None:
                eps = np.maximum(np.load(self.eps_file), EPS_MIN)
            else:
                eps = np.maximum(np.median(np.abs(omega), axis=0), EPS_MIN)
            w_raw = raw_weight(omega, eps, mode, c)
            norm = np.mean(w_raw, axis=0)
            loss_fn = InvIncrementWeightedMSE(t(Yscaler_mean), t(Yscaler_std), t(eps), t(norm), mode, c).to(self.device)
            cols = [str(c_) for c_ in self.Y_cols_all]
            self._log("WEIGHTED eps_k=" + ", ".join(f"{c_}={e:.3e}" for c_, e in zip(cols, eps)))
            self._log("WEIGHTED max/mean and min/mean weight per species (sample)=" + ", ".join(
                f"{c_}={mx:.2f}/{mn:.2e}" for c_, mx, mn in zip(cols, w_raw.max(axis=0) / norm, w_raw.min(axis=0) / norm)))

        hook = None
        if self.element_penalty > 0:
            if self.dt_var:
                raise NotImplementedError("element_penalty with dt_var")
            A = np.asarray(self.A_element, dtype=np.float64)
            A = A[np.abs(A).sum(axis=1) > 0]                # drop elements absent from the mechanism (e.g. C)
            holder = InputHolder()
            hook = model.register_forward_pre_hook(lambda m, inp: setattr(holder, "x", inp[0]))
            loss_fn = ElementPenaltyLoss(loss_fn, self, holder, t(A), t(Xscaler_mean), t(Xscaler_std), t(Yscaler_mean),
                                         t(Yscaler_std), self.element_penalty, self.element_drift_ref).to(self.device)
            self._log(f"ELEMENT_PENALTY on {A.shape[0]} elements, lambda={self.element_penalty}, ref={self.element_drift_ref}")
            self._drift_diag("before training", model, loss_fn, X_val)
            if scheduler is not None:
                step0 = scheduler.step
                count = [0]

                def step_and_log(*a, **k):
                    count[0] += 1
                    if count[0] % LOG_EVERY == 0 and loss_fn.acc[2] > 0:
                        b, p, n = loss_fn.acc
                        self._log(f"ELEMENT_PENALTY epoch={count[0]} mean base loss={b / n:.4e} mean penalty={p / n:.4e} "
                                  f"(lambda*penalty={self.element_penalty * p / n:.4e})")
                        loss_fn.acc[:] = [0.0, 0.0, 0]
                    return step0(*a, **k)
                scheduler.step = step_and_log

        out = super().train_model(i_cluster, model, loss_fn, optimizer, scheduler, X_train, X_val, Y_train, Y_val,
                                  Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std, Ymulti_train, Tmulti_train)
        if hook is not None:
            self._drift_diag("after training", model, loss_fn, X_val)
            hook.remove()
        return out
