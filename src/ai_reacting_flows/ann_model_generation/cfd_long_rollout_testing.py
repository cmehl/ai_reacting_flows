"""Long chemistry-only rollout of the ANN against CVODE on a random subsample of
a CFD snapshot's reacting cells (MPI).

Same recipe as FullDomainTester (same yaml, model loading, inference and
CVODE reference) but with ``nb_steps_long`` chained steps (default 100) on
``cell_fraction`` of the cells (default 3 %), keeping the whole trajectory of
every sampled cell so the analysis can follow how the error grows step by
step and which cells diverge.

Because a long chain can push a cell into a state the Cantera-based
temperature update cannot digest, the ANN rollout is guarded: a cell whose
prediction becomes non-finite, or whose temperature leaves [150, 6000] K, is
frozen at that bad state (recorded as such) and no longer advanced; a failure
inside a batch falls back to cell-by-cell so one bad cell does not lose its
neighbours.

Extra yaml keys (test_ann_vs_cvode.yaml): nb_steps_long, cell_fraction,
subsample_seed. Each rank writes ``<output_file stem>_partNNNN.h5``.
"""

import os

import numpy as np
import h5py
import oyaml as yaml

from ai_reacting_flows.ann_model_generation.cfd_full_domain_testing import FullDomainTester


class LongRolloutTester(FullDomainTester):

    T_MIN_OK = 150.0
    T_MAX_OK = 6000.0

    def __init__(self, run_folder: str | None = None):
        super().__init__(run_folder)
        with open(os.path.join(self.run_folder, "test_ann_vs_cvode.yaml"), "r") as file:
            params = yaml.safe_load(file)
        self.nb_steps = int(params.get("nb_steps_long", 100))
        self.cell_fraction = float(params.get("cell_fraction", 0.03))
        self.subsample_seed = int(params.get("subsample_seed", 0))

    # ------------------------------------------------------------------
    def _ann_rollout(self, T, P, Y, rank):
        n = T.shape[0]
        ns = self.n_species
        T_out = np.full((self.nb_steps, n), np.nan)
        Y_out = np.full((self.nb_steps, n, ns), np.nan)
        for c0 in range(0, n, self.chunk_size):
            c1 = min(c0 + self.chunk_size, n)
            Tc, Yc, Pc = T[c0:c1].copy(), Y[c0:c1].copy(), P[c0:c1]
            alive = np.ones(c1 - c0, dtype=bool)
            for k in range(self.nb_steps):
                if rank == 0 and k % 10 == 0:
                    print(f"[rank 0] ANN chunk {c0}/{n} step {k}", flush=True)
                idx = np.where(alive)[0]
                if idx.size:
                    try:
                        Tn, Yn, _ = self._predict_ann(Tc[idx], Pc[idx], Yc[idx])
                    except Exception:
                        Tn = np.full(idx.size, np.nan)
                        Yn = np.full((idx.size, ns), np.nan)
                        for a, i in enumerate(idx):
                            try:
                                t1, y1, _ = self._predict_ann(Tc[i:i + 1], Pc[i:i + 1], Yc[i:i + 1])
                                Tn[a], Yn[a] = t1[0], y1[0]
                            except Exception:
                                pass
                    Tc[idx], Yc[idx] = Tn, Yn
                    bad = (~np.isfinite(Tn) | (Tn < self.T_MIN_OK) | (Tn > self.T_MAX_OK)
                           | ~np.isfinite(Yn).all(axis=1))
                    alive[idx[bad]] = False
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
                  f"cell_fraction={self.cell_fraction}, T_threshold={self.T_threshold:g}K", flush=True)

        T0, P0, Y0, coords = self._read_block(i0, i1)
        idx_above = np.where(T0 > self.T_threshold)[0]
        rng = np.random.default_rng(self.subsample_seed + rank)
        sel = idx_above[rng.random(idx_above.size) < self.cell_fraction]
        T0, P0, Y0, coords = T0[sel], P0[sel], Y0[sel], coords[sel]
        print(f"[rank {rank}] {sel.size} sampled cells", flush=True)

        T_cv, Y_cv, failed = self._cvode_multi(T0, P0, Y0, rank)
        T_an, Y_an = self._ann_rollout(T0, P0, Y0, rank)

        stem, ext = os.path.splitext(self.output_file)
        with h5py.File(f"{stem}_part{rank:04d}{ext}", "w") as h5f:
            h5f.attrs["time_step"] = self.time_step
            h5f.attrs["nb_steps"] = self.nb_steps
            h5f.attrs["T_threshold"] = self.T_threshold
            h5f.attrs["test_data_file"] = self.test_data_file
            h5f.attrs["models_folder"] = self.models_folder
            h5f.attrs["species"] = self.species_names
            h5f.attrs["cell_fraction"] = self.cell_fraction
            h5f.create_dataset("coords", data=coords)
            h5f.create_dataset("T_ini", data=T0)
            h5f.create_dataset("Y_ini", data=Y0)
            h5f.create_dataset("T_cvode", data=T_cv)
            h5f.create_dataset("Y_cvode", data=Y_cv)
            h5f.create_dataset("T_ann", data=T_an)
            h5f.create_dataset("Y_ann", data=Y_an)
            h5f.create_dataset("cvode_failed", data=failed)

        comm.Barrier()
        if rank == 0:
            print(f">> Done, parts written as {stem}_part*.h5", flush=True)
