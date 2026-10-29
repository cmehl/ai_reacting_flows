"""LearningDatabase with one added method: physical radical-region
augmentation that ALSO fills Y_multi, so it works with the rollout
pipeline (the existing `_augment_radical_regions` only fills X/Y, single
step). Implemented as a subclass so database_processing.py itself is
untouched - same pattern as NN_manager_reg.py.

Reuses the existing, already-physical building blocks unchanged:
  - `_perturb_X` (log-normal multiplicative noise on the species mass
    fractions, clipped >=0, renormalized to sum to 1 - this IS the
    "physical" part: no copy-paste, every augmented row is a genuinely
    different, physically valid composition near an existing hard point).
  - the same Cantera mechanism (`self.detailed_mechanism`) and constant-
    pressure reactor type as the raw database builder and `_recompute_Y_cantera`.

What's new is only the *rollout* recompute: instead of a single
sim.advance(dt), the SAME reactor is advanced for self.nb_steps
consecutive dt's (mirrors build_raw_database_rollout.py's
_react_state_multi exactly), producing a real Y_multi block instead of
duplicating/faking one.

Also fixes a real mismatch for this project's CFD databases: the
species columns here are named plainly (e.g. "N2H2"), not "Y_N2H2" -
`_perturb_X`'s caller in `_augment_radical_regions` filters columns with
`startswith("Y_")`, which silently selects an EMPTY list for this naming
convention (so that method perturbs nothing here). This subclass derives
species_cols from self.col_names_Y instead (Temperature/Pressure/
Prog_var/HRR excluded, whatever is left is a species, in the exact order
the raw builder wrote them).
"""
import os

import h5py
import numpy as np
import pandas as pd
import cantera as ct

import ai_reacting_flows.databases_processing.database_processing as dbp


class LearningDatabaseAug(dbp.LearningDatabase):

    def __init__(self):
        super().__init__()
        # self.time_step is never set by the base class (a latent gap: only
        # exercised by the never-yet-run augment_database() path) - read it
        # straight from the raw rollout h5's own Y_multi.attrs["dt"], the
        # same value build_raw_database_rollout.py wrote it with.
        with h5py.File(os.path.join(self.dtb_folder, self.input_dtb_file), "r") as f:
            self.time_step = float(f["ITERATION_00000/Y_multi"].attrs["dt"])
        self._log(f"AUG: time_step={self.time_step:g}s (read from raw h5)")

    def _log(self, msg):
        print(f">> {msg}")

    def _perturb_X_local(self, X_batch, species_cols, noise_level):
        """Same physics as the inherited _perturb_X (log-normal multiplicative
        noise on the species mass fractions, clipped >=0, renormalized to sum
        to 1) - reimplemented locally because the inherited version's
        `Xp[species_cols].values *= ...` raises ValueError: output array is
        read-only under this pandas version's copy-on-write .values (a latent
        bug in the parent method, apparently never exercised until now; fixed
        here rather than touched there)."""
        Xp = X_batch.copy()
        Y = np.array(Xp[species_cols].to_numpy(), copy=True)
        Y = Y * np.exp(np.random.normal(0, noise_level, size=Y.shape))
        Y = np.clip(Y, 0.0, None)
        Y = Y / (np.sum(Y, axis=1, keepdims=True) + 1e-30)
        Xp[species_cols] = Y
        return Xp

    def _species_cols(self):
        exclude = {"Temperature", "Pressure", "Prog_var", "HRR", "cluster"}
        return [c for c in self.col_names_Y if c not in exclude]

    def _react_state_multi_local(self, T0, P0, Y0, gas):
        """Same physics as build_raw_database_rollout._react_state_multi:
        one continuing constant-pressure reactor, advanced self.nb_steps
        times, every intermediate state recorded."""
        gas.TPY = T0, P0, Y0
        r = ct.IdealGasConstPressureReactor(gas)
        sim = ct.ReactorNet([r])
        out_T = np.empty(self.nb_steps)
        out_Y = np.empty((self.nb_steps, len(gas.species_names)))
        for k in range(1, self.nb_steps + 1):
            sim.advance(k * self.time_step)
            out_T[k - 1] = gas.T
            out_Y[k - 1] = gas.Y
        return out_T, out_Y

    def augment_radical_regions_rollout(self, radical_species, quantile=0.95,
                                         T_min=700.0, T_max=2200.0, factor=3,
                                         batch_size=2000, noise_level=0.01):
        print(">> Starting radical-focused ROLLOUT augmentation")
        if not self.rollout:
            raise RuntimeError("augment_radical_regions_rollout requires rollout=true")

        X = self.X
        species_cols = self._species_cols()
        print(f"   species columns used for perturbation ({len(species_cols)}): {species_cols}")

        mask = (X["Temperature"] > T_min) & (X["Temperature"] < T_max)
        radical_mask = np.zeros(len(X), dtype=bool)
        for sp in radical_species:
            if sp not in X.columns:
                print(f"   WARNING: {sp} not in X.columns, skipped")
                continue
            threshold = X[sp].quantile(quantile)
            hit = X[sp] > threshold
            radical_mask |= hit
            print(f"   {sp}: threshold(q={quantile})={threshold:.3e}, {int(hit.sum())} points")
        mask &= radical_mask

        X_target = X[mask].copy()
        n_target = len(X_target)
        print(f">> Selected {n_target} radical points in [{T_min},{T_max}] K, factor={factor} "
              f"-> {n_target * factor} new physical rollout samples")
        if n_target == 0:
            print(">> Nothing to augment, returning unchanged")
            return

        gas = ct.Solution(self.detailed_mechanism)
        col_order = list(self.col_names_Y)
        n_cols_Y = len(col_order)
        i_T, i_P = col_order.index("Temperature"), col_order.index("Pressure")
        i_sp = [col_order.index(c) for c in species_cols]
        i_extra = [col_order.index(c) for c in ("Prog_var", "HRR") if c in col_order]

        X_aug_all, Ymulti_aug_all = [], []
        n_batches = int(np.ceil(n_target / batch_size))
        for b in range(n_batches):
            i0, i1 = b * batch_size, min((b + 1) * batch_size, n_target)
            X_batch = X_target.iloc[i0:i1]
            print(f"   batch {b+1}/{n_batches} ({len(X_batch)} pts)")
            for _ in range(factor):
                X_pert = self._perturb_X_local(X_batch, species_cols, noise_level)   # physical noise
                n = len(X_pert)
                Ym = np.empty((n, self.nb_steps, n_cols_Y))
                if i_extra:
                    Ym[:, :, i_extra] = -1.0
                for j in range(n):
                    row = X_pert.iloc[j]
                    T_traj, Y_traj = self._react_state_multi_local(
                        row["Temperature"], row["Pressure"], row[species_cols].values, gas
                    )
                    Ym[j, :, i_T] = T_traj
                    Ym[j, :, i_P] = row["Pressure"]  # constant-pressure reactor
                    Ym[j, :, i_sp] = Y_traj.T
                X_aug_all.append(X_pert)
                Ymulti_aug_all.append(Ym)

        X_aug = pd.concat(X_aug_all, ignore_index=True)
        Ymulti_aug = np.concatenate(Ymulti_aug_all, axis=0)
        Y_aug = pd.DataFrame(Ymulti_aug[:, 0, :], columns=col_order)

        self.X = pd.concat([self.X, X_aug], ignore_index=True)
        self.Y = pd.concat([self.Y, Y_aug], ignore_index=True)
        self.Y_multi = np.concatenate([self.Y_multi, Ymulti_aug], axis=0)

        print(f">> Rollout radical augmentation completed: +{len(X_aug)} rows, new size = {len(self.X)}")
