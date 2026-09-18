"""Full-domain, multi-step ANN vs CVODE comparison on a CFD snapshot (MPI).

Extends CFDSnapshotTester (same yaml, same model/scaler loading, same
inference recipe via ``_predict_ann``) from "a slice or a random sample, one
step, serial" to "every cell of the snapshot, ``nb_steps`` chained steps,
MPI-parallel":

- CVODE reference: one ``IdealGasConstPressureReactor`` per cell advanced to
  k*dt for k=1..nb_steps (same recipe as build_raw_database_rollout.py, so the
  reference is exactly the ground truth the network was trained against).
- ANN: autoregressive rollout from the same initial state, feeding each
  step's prediction (Y as predicted, Temperature from enthalpy conservation)
  back in -- the deployed behaviour.

Cells at or below T_threshold are left untouched on both sides (identity).

Each rank handles a contiguous block of cells and writes its own part file
``<output_file stem>_part<rank>.h5`` (no gather of ~10 GB through MPI); the
analysis script concatenates the parts.

Run (from a folder holding test_ann_vs_cvode.yaml and MODELS/):
    mpirun -n <nb_procs> python test_full_domain.py
"""

import os

import numpy as np
import h5py
import cantera as ct
import torch

from ai_reacting_flows.ann_model_generation.cfd_snapshot_testing import CFDSnapshotTester


class FullDomainTester(CFDSnapshotTester):

    def __init__(self, run_folder: str | None = None):
        super().__init__(run_folder)

        import oyaml as yaml
        with open(os.path.join(self.run_folder, "test_ann_vs_cvode.yaml"), "r") as file:
            params = yaml.safe_load(file)
        self.nb_steps = int(params.get("nb_steps", 3))
        self.chunk_size = int(params.get("chunk_size", 50000))

        torch.set_num_threads(1)

    # ------------------------------------------------------------------
    def _read_block(self, i0, i1):
        with h5py.File(self.test_data_file, "r") as f:
            g = f["STREAM_00/CELL_CENTER_DATA"]
            T = g["TEMPERATURE"][i0:i1].astype(np.float64)
            P = g["PRESSURE"][i0:i1].astype(np.float64)
            coords = np.stack([g["XCEN_X"][i0:i1], g["XCEN_Y"][i0:i1], g["XCEN_Z"][i0:i1]], axis=1).astype(np.float64)
            Y = np.empty((T.shape[0], self.n_species))
            for j, name in enumerate(self.species_names):
                Y[:, j] = g[f"MASSFRAC_{name}"][i0:i1]
        Y = Y / Y.sum(axis=1, keepdims=True)
        return T, P, Y, coords

    def _cvode_multi(self, T, P, Y, rank):
        """[nb_steps, n] T and [nb_steps, n, ns] Y, chained on one reactor per cell."""
        n = T.shape[0]
        T_out = np.empty((self.nb_steps, n))
        Y_out = np.empty((self.nb_steps, n, self.n_species))
        failed = np.zeros(n, dtype=bool)
        for i in range(n):
            if rank == 0 and i % 20000 == 0:
                print(f"[rank 0] CVODE {i} / {n}", flush=True)
            try:
                self.gas.TPY = T[i], P[i], Y[i]
                r = ct.IdealGasConstPressureReactor(self.gas)
                sim = ct.ReactorNet([r])
                for k in range(self.nb_steps):
                    sim.advance((k + 1) * self.time_step)
                    T_out[k, i] = self.gas.T
                    Y_out[k, i] = self.gas.Y
            except ct.CanteraError:
                failed[i] = True
                T_out[:, i] = T[i]
                Y_out[:, i] = Y[i]
        return T_out, Y_out, failed

    def _ann_rollout(self, T, P, Y, rank):
        n = T.shape[0]
        T_out = np.empty((self.nb_steps, n))
        Y_out = np.empty((self.nb_steps, n, self.n_species))
        for c0 in range(0, n, self.chunk_size):
            c1 = min(c0 + self.chunk_size, n)
            if rank == 0:
                print(f"[rank 0] ANN {c0} / {n}", flush=True)
            Tc, Yc, Pc = T[c0:c1].copy(), Y[c0:c1].copy(), P[c0:c1]
            for k in range(self.nb_steps):
                Tc, Yc, _ = self._predict_ann(Tc, Pc, Yc)
                T_out[k, c0:c1] = Tc
                Y_out[k, c0:c1] = Yc
        return T_out, Y_out

    # ------------------------------------------------------------------
    def run_mpi(self, comm):
        rank, size = comm.Get_rank(), comm.Get_size()

        with h5py.File(self.test_data_file, "r") as f:
            n_total = f["STREAM_00/CELL_CENTER_DATA/TEMPERATURE"].shape[0]
        bounds = np.linspace(0, n_total, size + 1).astype(int)
        i0, i1 = int(bounds[rank]), int(bounds[rank + 1])

        if rank == 0:
            print(f">> {n_total} cells, {size} rank(s), nb_steps={self.nb_steps}, dt={self.time_step:g}s, "
                  f"T_threshold={self.T_threshold:g}K", flush=True)

        T0, P0, Y0, coords = self._read_block(i0, i1)
        n = T0.shape[0]
        above = T0 > self.T_threshold
        idx_above = np.where(above)[0]

        T_cv = np.repeat(T0[None, :], self.nb_steps, axis=0)
        Y_cv = np.repeat(Y0[None, :, :], self.nb_steps, axis=0)
        T_an, Y_an = T_cv.copy(), Y_cv.copy()
        failed = np.zeros(n, dtype=bool)

        if idx_above.size:
            Tc, Yc, fl = self._cvode_multi(T0[idx_above], P0[idx_above], Y0[idx_above], rank)
            T_cv[:, idx_above], Y_cv[:, idx_above], failed[idx_above] = Tc, Yc, fl
            Ta, Ya = self._ann_rollout(T0[idx_above], P0[idx_above], Y0[idx_above], rank)
            T_an[:, idx_above], Y_an[:, idx_above] = Ta, Ya

        stem, ext = os.path.splitext(self.output_file)
        part_path = f"{stem}_part{rank:04d}{ext}"
        with h5py.File(part_path, "w") as h5f:
            h5f.attrs["time_step"] = self.time_step
            h5f.attrs["nb_steps"] = self.nb_steps
            h5f.attrs["T_threshold"] = self.T_threshold
            h5f.attrs["test_data_file"] = self.test_data_file
            h5f.attrs["models_folder"] = self.models_folder
            h5f.attrs["species"] = self.species_names
            h5f.attrs["cell_start"] = i0
            h5f.create_dataset("coords", data=coords)
            h5f.create_dataset("T_ini", data=T0)
            h5f.create_dataset("P_ini", data=P0)
            h5f.create_dataset("Y_ini", data=Y0)
            h5f.create_dataset("T_cvode", data=T_cv)
            h5f.create_dataset("Y_cvode", data=Y_cv)
            h5f.create_dataset("T_ann", data=T_an)
            h5f.create_dataset("Y_ann", data=Y_an)
            h5f.create_dataset("above", data=above)
            h5f.create_dataset("cvode_failed", data=failed)

        comm.Barrier()
        if rank == 0:
            print(f">> Done, parts written as {stem}_part*.h5", flush=True)
