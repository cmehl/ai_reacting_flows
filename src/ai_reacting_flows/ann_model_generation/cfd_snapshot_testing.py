"""Single-step ANN vs CVODE comparison on a held-out CFD snapshot.

Reads one raw CFD field snapshot (same HDF5 layout as
``DATA/REDUCED/output/*.h5`` used by ``build_raw_database.py``), samples a
subset of cells above ``T_threshold``, and advances each sampled cell by
``time_step`` two ways:

- CVODE: ``cantera.IdealGasConstPressureReactor`` (ground truth).
- ANN: the trained per-cluster MLP model(s) for ``models_folder``, using the
  exact same cluster-assignment / normalization / reconstruction recipe as
  database_processing.py + NN_manager.py, so results reflect the model as it
  is actually trained and would be deployed.

Results (initial state, CVODE state, ANN state, cluster id) are written to a
single HDF5 file for later analysis in a notebook.
"""

import os
import glob

import numpy as np
import h5py
import oyaml as yaml
import cantera as ct

import torch


class CFDSnapshotTester:

    def __init__(self, run_folder: str | None = None):

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.run_folder = os.path.abspath(run_folder) if run_folder is not None else os.getcwd()

        with open(os.path.join(self.run_folder, "test_ann_vs_cvode.yaml"), "r") as file:
            test_params = yaml.safe_load(file)

        self.test_data_file = os.path.join(self.run_folder, test_params["test_data_file"])
        self.time_step = float(test_params["time_step"])
        self.T_threshold = float(test_params["T_threshold"])
        self.n_sample = int(test_params["n_sample"])
        self.seed = int(test_params.get("seed", 0))
        self.output_file = os.path.join(self.run_folder, test_params["output_file"])

        # --- Model folder / training configuration -------------------------
        self.models_folder = os.path.join(self.run_folder, "MODELS", test_params["models_folder"])
        with open(os.path.join(self.models_folder, "networks_params.yaml"), "r") as file:
            networks_parameters = yaml.safe_load(file)
        self.dataset_path = os.path.join(self.run_folder, networks_parameters["database_path"])

        with open(os.path.join(self.dataset_path, "dtb_processing.yaml"), "r") as file:
            dtb_processing_params = yaml.safe_load(file)

        data_processing = dtb_processing_params["data_processing"]
        self.log_transform_X = data_processing["log_transform_X"]
        self.log_transform_Y = data_processing["log_transform_Y"]
        self.lambda_bct = data_processing.get("lambda_bct", 0.1)
        self.threshold = data_processing["threshold"]
        self.output_omegas = data_processing["output_omegas"]
        self.with_N_chemistry = data_processing["with_N_chemistry"]
        self.log_excluded_species = data_processing.get("log_excluded_species", [])

        data_clustering = dtb_processing_params["data_clustering"]
        self.clustering_method_cfg = data_clustering["clustering_method"]
        self.nb_clusters = data_clustering["nb_clusters"]

        # --- Mechanism: reuse the copy NN_manager placed in the model folder
        mech_candidates = [
            p for p in glob.glob(os.path.join(self.models_folder, "*.yaml"))
            if os.path.basename(p) != "networks_params.yaml"
        ]
        if len(mech_candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one mechanism YAML copy in {self.models_folder}, found {mech_candidates}"
            )
        self.mech_file = mech_candidates[0]
        self.gas = ct.Solution(self.mech_file)
        self.species_names = self.gas.species_names
        self.n_species = len(self.species_names)

        if not self.with_N_chemistry and "N2" in self.species_names:
            raise NotImplementedError(
                "with_N_chemistry=false (N2 removed from ANN state) is not handled by "
                "CFDSnapshotTester yet; this model keeps N2 so it is out of scope."
            )

        # --- Species/column consistency check against training data --------
        with h5py.File(os.path.join(self.dataset_path, "training_data.h5"), "r") as h5file_r:
            x_cols = [str(c) for c in h5file_r["CLUSTER_0/X_train"].attrs["cols"]]
            y_cols = [str(c) for c in h5file_r["CLUSTER_0/Y_train"].attrs["cols"]]

        expected_x = ["Temperature_X"] + [f"{s}_X" for s in self.species_names]
        expected_y = [f"{s}_Y" for s in self.species_names]
        if x_cols != expected_x or y_cols != expected_y:
            raise ValueError(
                "Species order/columns in training_data.h5 do not match "
                f"gas.species_names from {self.mech_file}.\n"
                f"training X cols: {x_cols}\nexpected: {expected_x}"
            )

        # --- Load per-cluster models and scalers ----------------------------
        self.nb_clusters = len(glob.glob(os.path.join(self.models_folder, "cluster*_model.pth")))
        if self.nb_clusters != data_clustering["nb_clusters"]:
            raise ValueError(
                f"nb_clusters mismatch: {self.nb_clusters} model file(s) found in "
                f"{self.models_folder} but dtb_processing.yaml says {data_clustering['nb_clusters']}"
            )

        self.models = []
        self.Xscaler_mean, self.Xscaler_std = [], []
        self.Yscaler_mean, self.Yscaler_std = [], []
        with h5py.File(os.path.join(self.dataset_path, "training_data.h5"), "r") as h5file_r:
            for i in range(self.nb_clusters):
                model = torch.load(
                    os.path.join(self.models_folder, f"cluster{i}_model.pth"), weights_only=False
                )
                model.to(self.device)
                model.eval()
                self.models.append(model)

                grp = h5file_r[f"CLUSTER_{i}"]
                Xscaler_array = grp["Xscaler"][:]
                Yscaler_array = grp["Yscaler"][:]
                self.Xscaler_mean.append(Xscaler_array[:, 0])
                self.Xscaler_std.append(np.sqrt(Xscaler_array[:, 1]))
                self.Yscaler_mean.append(Yscaler_array[:, 0])
                self.Yscaler_std.append(np.sqrt(Yscaler_array[:, 1]))

        # --- Load k-means clustering artifacts if relevant -------------------
        if self.nb_clusters > 1:
            import pickle
            import joblib

            with open(os.path.join(self.models_folder, "kmeans_model.pkl"), "rb") as f:
                self.kmeans = pickle.load(f)
            self.kmeans_scaler = joblib.load(os.path.join(self.models_folder, "Xscaler_kmeans.pkl"))

    # ------------------------------------------------------------------
    # CFD snapshot reading (mirrors build_raw_database.py._read_cfd_file)
    # ------------------------------------------------------------------
    def _read_cfd_snapshot(self):

        with h5py.File(self.test_data_file, "r") as f:
            cell_data = f["STREAM_00/CELL_CENTER_DATA"]

            T = cell_data["TEMPERATURE"][()].astype(np.float64)
            P = cell_data["PRESSURE"][()].astype(np.float64)

            Y = np.empty((T.shape[0], self.n_species), dtype=np.float64)
            for i_sp, name in enumerate(self.species_names):
                key = f"MASSFRAC_{name}"
                if key not in cell_data:
                    raise KeyError(f"Species '{name}' (dataset '{key}') not found in {self.test_data_file}")
                Y[:, i_sp] = cell_data[key][()].astype(np.float64)

        return T, P, Y

    def _sample_cells(self, T, P, Y):

        mask = T > self.T_threshold
        T, P, Y = T[mask], P[mask], Y[mask]

        rng = np.random.default_rng(self.seed)
        n = min(self.n_sample, T.shape[0])
        idx = rng.choice(T.shape[0], size=n, replace=False)
        T, P, Y = T[idx], P[idx], Y[idx]

        # CFD post-processing mass fractions rarely sum exactly to 1.
        Y = Y / Y.sum(axis=1, keepdims=True)

        return T, P, Y

    # ------------------------------------------------------------------
    # CVODE reference
    # ------------------------------------------------------------------
    def _react_cvode(self, T, P, Y):

        n = T.shape[0]
        T_new = np.empty(n)
        Y_new = np.empty_like(Y)

        for i in range(n):
            if i % 500 == 0:
                print(f"  CVODE {i} / {n}", flush=True)

            self.gas.TPY = T[i], P[i], Y[i]
            r = ct.IdealGasConstPressureReactor(self.gas)
            sim = ct.ReactorNet([r])
            sim.advance(self.time_step)

            T_new[i] = self.gas.T
            Y_new[i] = self.gas.Y

        return T_new, Y_new

    # ------------------------------------------------------------------
    # ANN prediction
    # ------------------------------------------------------------------
    def _inverse_transform_cols(self, transformed, log_transform):
        """Undo a per-column log/Box-Cox transform, skipping log_excluded_species.

        `transformed` has one column per species, in self.species_names order.
        """

        out = transformed.copy()
        if log_transform == 0:
            return out

        for j, name in enumerate(self.species_names):
            if name in self.log_excluded_species:
                continue
            if log_transform == 1:
                out[:, j] = np.exp(transformed[:, j])
            elif log_transform == 2:
                out[:, j] = (transformed[:, j] * self.lambda_bct + 1.0) ** (1.0 / self.lambda_bct)

        return out

    def _assign_clusters(self, T, P, Y):

        n = T.shape[0]
        if self.nb_clusters == 1:
            return np.zeros(n, dtype=int)

        # Feature vector exactly as actually fitted by
        # LearningDatabase.clusterize_dataset for dt_var=False (kmeans branch):
        # raw = [Temperature, Pressure, species..., Prog_var(-1), HRR(-1), cluster(0)],
        # then log-transform is applied to columns [1, n_species] inclusive
        # (i.e. Pressure + all species except the last one) because that code
        # indexes into this raw layout using offsets meant for a [T, species]
        # vector. This mismatch is a pre-existing bug (see NN_manager fix
        # discussion); it is reproduced here on purpose so cells are routed to
        # the same cluster/model the training pipeline actually used.
        raw = np.column_stack([T, P, Y, -np.ones(n), -np.ones(n), np.zeros(n)])

        if self.log_transform_X > 0:
            log_cols = list(range(1, 1 + self.n_species))
            raw[:, log_cols] = np.clip(raw[:, log_cols], self.threshold, None)
            if self.log_transform_X == 1:
                raw[:, log_cols] = np.log(raw[:, log_cols])
            elif self.log_transform_X == 2:
                raw[:, log_cols] = (raw[:, log_cols] ** self.lambda_bct - 1.0) / self.lambda_bct

        scaled = self.kmeans_scaler.transform(raw)
        return self.kmeans.predict(scaled)

    def _predict_ann(self, T, P, Y):

        n = T.shape[0]
        cluster_labels = self._assign_clusters(T, P, Y)

        Y_new = np.empty_like(Y)
        T_new = np.empty(n)

        for c in range(self.nb_clusters):
            mask = cluster_labels == c
            if not mask.any():
                continue

            Tc = T[mask]
            Yc = Y[mask]

            # Transform species part of X (clip + log/BCT), respecting log_excluded_species.
            Xc_species_t = Yc.copy()
            if self.log_transform_X > 0:
                for j, name in enumerate(self.species_names):
                    if name in self.log_excluded_species:
                        continue
                    val = np.clip(Yc[:, j], self.threshold, None)
                    if self.log_transform_X == 1:
                        Xc_species_t[:, j] = np.log(val)
                    elif self.log_transform_X == 2:
                        Xc_species_t[:, j] = (val ** self.lambda_bct - 1.0) / self.lambda_bct

            Xc_full = np.column_stack([Tc, Xc_species_t])
            Xc_scaled = (Xc_full - self.Xscaler_mean[c]) / (self.Xscaler_std[c] + 1e-7)

            with torch.no_grad():
                inp = torch.tensor(Xc_scaled, dtype=torch.float64, device=self.device)
                pred = self.models[c](inp).detach().cpu().numpy()

            # Affine-unscale only (still "transformed" space): mirrors
            # NN_manager._inverse_scale's first line, without exponentiating yet.
            omega_t = self.Yscaler_mean[c] + (self.Yscaler_std[c] + 1e-7) * pred

            if self.output_omegas and self.log_transform_Y > 0:
                # omega_t approximates log(Y)-log(X) (or the BCT equivalent): must
                # recombine in transformed space BEFORE exponentiating/BCT-inverting.
                combined = Xc_species_t + omega_t
                Yc_new = self._inverse_transform_cols(combined, self.log_transform_Y)
            else:
                yk = self._inverse_transform_cols(omega_t, self.log_transform_Y)
                Yc_new = Yc + yk if self.output_omegas else yk

            Y_new[mask] = Yc_new

        # Temperature from enthalpy conservation (Cantera is not vectorizable,
        # so this stays a per-cell loop, but it's cheap: no ODE integration).
        for i in range(n):
            self.gas.TPY = T[i], P[i], Y[i]
            cp = self.gas.cp
            h_molar = self.gas.partial_molar_enthalpies
            mw = self.gas.molecular_weights
            T_new[i] = T[i] - (1.0 / cp) * np.sum(h_molar / mw * (Y_new[i] - Y[i]))

        return T_new, Y_new, cluster_labels

    # ------------------------------------------------------------------
    def run(self):

        print(f">> Reading test snapshot {self.test_data_file}")
        T_all, P_all, Y_all = self._read_cfd_snapshot()

        print(f">> Sampling {self.n_sample} cells above T_threshold={self.T_threshold:g} K")
        T0, P0, Y0 = self._sample_cells(T_all, P_all, Y_all)
        n = T0.shape[0]
        print(f"   >> {n} cells sampled")

        print(f">> Reacting with CVODE (dt={self.time_step:g} s)")
        T_cvode, Y_cvode = self._react_cvode(T0, P0, Y0)

        print(">> Predicting with ANN")
        T_ann, Y_ann, cluster_labels = self._predict_ann(T0, P0, Y0)

        print(f">> Writing results to {self.output_file}")
        with h5py.File(self.output_file, "w") as h5f:
            h5f.attrs["time_step"] = self.time_step
            h5f.attrs["T_threshold"] = self.T_threshold
            h5f.attrs["test_data_file"] = self.test_data_file
            h5f.attrs["models_folder"] = self.models_folder
            h5f.attrs["species"] = self.species_names

            h5f.create_dataset("T_ini", data=T0)
            h5f.create_dataset("P_ini", data=P0)
            h5f.create_dataset("Y_ini", data=Y0)
            h5f.create_dataset("T_cvode", data=T_cvode)
            h5f.create_dataset("Y_cvode", data=Y_cvode)
            h5f.create_dataset("T_ann", data=T_ann)
            h5f.create_dataset("Y_ann", data=Y_ann)
            h5f.create_dataset("cluster", data=cluster_labels)

        print(">> Done")
