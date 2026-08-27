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
import pandas as pd
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

        self.time_step = float(test_params["time_step"])
        self.T_threshold = float(test_params["T_threshold"])
        # Opt-in diagnostic mode: also skip CVODE (identity) below
        # T_threshold, instead of always reacting every cell. Default False
        # preserves the reference-deployed-behavior mirroring documented on
        # _react_cvode/_predict_ann -- only turn this on to isolate genuine
        # model error from the below-threshold masking artifact (cells where
        # CVODE reacts but the ANN is skipped by design).
        self.mask_cvode_below_threshold = bool(test_params.get("mask_cvode_below_threshold", False))
        self.seed = int(test_params.get("seed", 0))
        self.output_file = os.path.join(self.run_folder, test_params["output_file"])

        # --- Cell selection: one of three modes --------------------------
        # 1. csv_slices: pre-cut slices (e.g. exported from ParaView), one CSV
        #    per slice, every row used as-is (no thickness/sampling -- the
        #    cut is already exact). Preferred whenever available: a real
        #    plane cut through the mesh, not an approximation via a thin 3D
        #    slab that can under/over-sample depending on local mesh density.
        # 2. slices: a few fixed thin 3D slabs cut programmatically from a
        #    full raw snapshot (coordinates kept, for spatial comparison).
        # 3. neither: historical random sample of n_sample cells.
        self.csv_slices = test_params.get("csv_slices")
        self.slices = test_params.get("slices")
        self.test_data_file = (
            os.path.join(self.run_folder, test_params["test_data_file"])
            if not self.csv_slices else None
        )
        self.n_sample = (
            int(test_params["n_sample"]) if not (self.csv_slices or self.slices) else None
        )

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
                    os.path.join(self.models_folder, f"cluster{i}_model.pth"),
                    map_location=self.device,
                    weights_only=False,
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
            coord_x = cell_data["XCEN_X"][()].astype(np.float64)
            coord_y = cell_data["XCEN_Y"][()].astype(np.float64)
            coord_z = cell_data["XCEN_Z"][()].astype(np.float64)

            Y = np.empty((T.shape[0], self.n_species), dtype=np.float64)
            for i_sp, name in enumerate(self.species_names):
                key = f"MASSFRAC_{name}"
                if key not in cell_data:
                    raise KeyError(f"Species '{name}' (dataset '{key}') not found in {self.test_data_file}")
                Y[:, i_sp] = cell_data[key][()].astype(np.float64)

        return T, P, Y, coord_x, coord_y, coord_z

    # ------------------------------------------------------------------
    # CSV slice reading (pre-cut slices, e.g. exported from ParaView)
    # ------------------------------------------------------------------
    def _read_csv_slice(self, csv_path):

        df = pd.read_csv(csv_path)

        T = df["TEMPERATURE"].to_numpy(dtype=np.float64)
        P = df["PRESSURE"].to_numpy(dtype=np.float64)
        coord_x = df["XCEN_0"].to_numpy(dtype=np.float64)
        coord_y = df["XCEN_1"].to_numpy(dtype=np.float64)
        coord_z = df["XCEN_2"].to_numpy(dtype=np.float64)

        Y = np.empty((T.shape[0], self.n_species), dtype=np.float64)
        for i_sp, name in enumerate(self.species_names):
            key = f"MASSFRAC_{name}"
            if key not in df.columns:
                raise KeyError(f"Species '{name}' (column '{key}') not found in {csv_path}")
            Y[:, i_sp] = df[key].to_numpy(dtype=np.float64)

        # No T_threshold masking here: every cell is kept and passed through
        # to _react_cvode/_predict_ann, which mirror the reference behavior
        # in NN_testing.run_1D_premixed_case -- CVODE always reacts every
        # cell (even cold/inert ones, at negligible cost), while the ANN
        # side only predicts for T >= T_threshold and otherwise returns the
        # state unchanged (identity), rather than dropping the cell
        # entirely. Masking here would have silently invented a third,
        # wrong behavior (cell just doesn't exist in the output).

        # CFD post-processing mass fractions rarely sum exactly to 1.
        Y = Y / Y.sum(axis=1, keepdims=True)

        return T, P, Y, coord_x, coord_y, coord_z

    def _run_csv_slices(self):

        T_l, P_l, Y_l, X_l, Yc_l, Z_l, sid_l, slices_meta = [], [], [], [], [], [], [], []

        for i, sl in enumerate(self.csv_slices):
            csv_path = os.path.join(self.run_folder, sl["file"])
            print(f">> Reading CSV slice {csv_path}")
            T, P, Y, cx, cy, cz = self._read_csv_slice(csv_path)
            n = T.shape[0]
            print(f"   >> {n} cells ({int((T >= self.T_threshold).sum())} >= T_threshold={self.T_threshold:g} K)")

            T_l.append(T); P_l.append(P); Y_l.append(Y)
            X_l.append(cx); Yc_l.append(cy); Z_l.append(cz)
            sid_l.append(np.full(n, i, dtype=int))
            slices_meta.append({
                "axis": str(sl["axis"]), "center": float(sl["center"]), "thickness": 0.0,
            })

        T0, P0, Y0 = np.concatenate(T_l), np.concatenate(P_l), np.concatenate(Y_l)
        X0, Ycoord0, Z0 = np.concatenate(X_l), np.concatenate(Yc_l), np.concatenate(Z_l)
        slice_id = np.concatenate(sid_l)
        n = T0.shape[0]
        print(f">> {n} cells total across {len(self.csv_slices)} CSV slice(s)")

        print(f">> Reacting with CVODE (dt={self.time_step:g} s)")
        T_cvode, Y_cvode = self._react_cvode(T0, P0, Y0)

        print(">> Predicting with ANN")
        T_ann, Y_ann, cluster_labels = self._predict_ann(T0, P0, Y0)

        print(f">> Writing results to {self.output_file}")
        with h5py.File(self.output_file, "w") as h5f:
            h5f.attrs["time_step"] = self.time_step
            h5f.attrs["T_threshold"] = self.T_threshold
            h5f.attrs["test_data_file"] = ";".join(sl["file"] for sl in self.csv_slices)
            h5f.attrs["models_folder"] = self.models_folder
            h5f.attrs["species"] = self.species_names
            h5f.attrs["slices"] = yaml.dump(slices_meta)

            h5f.create_dataset("T_ini", data=T0)
            h5f.create_dataset("P_ini", data=P0)
            h5f.create_dataset("Y_ini", data=Y0)
            h5f.create_dataset("T_cvode", data=T_cvode)
            h5f.create_dataset("Y_cvode", data=Y_cvode)
            h5f.create_dataset("T_ann", data=T_ann)
            h5f.create_dataset("Y_ann", data=Y_ann)
            h5f.create_dataset("cluster", data=cluster_labels)
            h5f.create_dataset("XCEN_X", data=X0)
            h5f.create_dataset("XCEN_Y", data=Ycoord0)
            h5f.create_dataset("XCEN_Z", data=Z0)
            h5f.create_dataset("slice_id", data=slice_id)

        print(">> Done")

    def _select_cells(self, T, P, Y, coord_x, coord_y, coord_z):

        mask = T > self.T_threshold

        # slice_id[i] = index of the slice (in self.slices) cell i was picked
        # from, or -1 when slices aren't used (random-sample path).
        if self.slices:
            coords = {"X": coord_x, "Y": coord_y, "Z": coord_z}
            slice_id_full = np.full(T.shape[0], -1, dtype=int)
            for i, sl in enumerate(self.slices):
                axis_coord = coords[str(sl["axis"]).upper()]
                half_thickness = 0.5 * float(sl["thickness"])
                in_slice = np.abs(axis_coord - float(sl["center"])) <= half_thickness
                # first matching slice wins for cells that fall in an overlap
                slice_id_full[in_slice & (slice_id_full == -1)] = i
            mask &= slice_id_full >= 0
            slice_id = slice_id_full[mask]
        else:
            slice_id = np.full(mask.sum(), -1, dtype=int)

        T, P, Y = T[mask], P[mask], Y[mask]
        coord_x, coord_y, coord_z = coord_x[mask], coord_y[mask], coord_z[mask]

        if not self.slices:
            rng = np.random.default_rng(self.seed)
            n = min(self.n_sample, T.shape[0])
            idx = rng.choice(T.shape[0], size=n, replace=False)
            T, P, Y = T[idx], P[idx], Y[idx]
            coord_x, coord_y, coord_z = coord_x[idx], coord_y[idx], coord_z[idx]
            slice_id = slice_id[idx]

        # CFD post-processing mass fractions rarely sum exactly to 1.
        Y = Y / Y.sum(axis=1, keepdims=True)

        return T, P, Y, coord_x, coord_y, coord_z, slice_id

    # ------------------------------------------------------------------
    # CVODE reference
    # ------------------------------------------------------------------
    def _react_cvode(self, T, P, Y):

        n = T.shape[0]
        T_new = np.empty(n)
        Y_new = np.empty_like(Y)
        n_failed = 0
        n_masked = 0

        for i in range(n):
            if i % 500 == 0:
                print(f"  CVODE {i} / {n}", flush=True)

            if self.mask_cvode_below_threshold and T[i] < self.T_threshold:
                # Diagnostic mode: mirror the ANN's own masking so cells
                # below T_threshold don't react on either side, isolating
                # genuine model error from the masking artifact.
                n_masked += 1
                T_new[i] = T[i]
                Y_new[i] = Y[i]
                continue

            try:
                self.gas.TPY = T[i], P[i], Y[i]
                r = ct.IdealGasConstPressureReactor(self.gas)
                sim = ct.ReactorNet([r])
                sim.advance(self.time_step)

                T_new[i] = self.gas.T
                Y_new[i] = self.gas.Y
            except ct.CanteraError as exc:
                # Genuinely degenerate/edge-case CFD post-processing states
                # (typically at/near the cold ambient inlet) can be too stiff
                # for CVODE to converge even though nothing is really
                # reacting there. Fall back to identity for that one cell
                # rather than aborting the whole run -- consistent with the
                # "negligible reaction -> unchanged state" principle already
                # applied on the ANN side below T_threshold.
                n_failed += 1
                print(f"  CVODE cell {i}: integration failed ({exc.__class__.__name__}), "
                      f"falling back to identity. T={T[i]:.1f}K", flush=True)
                T_new[i] = T[i]
                Y_new[i] = Y[i]

        if n_masked:
            print(f"  CVODE: {n_masked}/{n} cell(s) skipped (identity) below T_threshold={self.T_threshold:g}K, mask_cvode_below_threshold=true", flush=True)
        if n_failed:
            print(f"  CVODE: {n_failed}/{n} cell(s) fell back to identity after a solver failure", flush=True)

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

        # Feature vector as fitted by LearningDatabase.clusterize_dataset
        # (kmeans branch, dt_var=False): [Temperature, species...], with the
        # log/BCT transform applied to the species columns only (Temperature
        # untouched), respecting log_excluded_species. Must stay in lockstep
        # with database_processing.py's clusterize_dataset -- see branch
        # fix/kmeans-cluster-features. with_N_chemistry=false (N2 dropped) is
        # rejected in __init__, so N2 is always present here.
        raw = np.column_stack([T, Y])

        if self.log_transform_X > 0:
            for j, name in enumerate(self.species_names):
                if name in self.log_excluded_species:
                    continue
                col = 1 + j
                val = np.clip(raw[:, col], self.threshold, None)
                if self.log_transform_X == 1:
                    raw[:, col] = np.log(val)
                elif self.log_transform_X == 2:
                    raw[:, col] = (val ** self.lambda_bct - 1.0) / self.lambda_bct

        scaled = self.kmeans_scaler.transform(raw)
        return self.kmeans.predict(scaled)

    def _predict_ann(self, T, P, Y):

        # Mirrors NN_testing.run_1D_premixed_case: below T_threshold the ANN
        # is not queried at all -- the state is left unchanged (identity),
        # exactly like the reference deployed behavior -- rather than being
        # dropped or, worse, fed to a network trained only on T >= threshold
        # states (undefined behavior on out-of-distribution cold inputs).
        n = T.shape[0]
        above = T >= self.T_threshold

        cluster_labels = np.full(n, -1, dtype=int)
        if above.any():
            cluster_labels[above] = self._assign_clusters(T[above], P[above], Y[above])

        Y_new = Y.copy()
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

        if self.csv_slices:
            self._run_csv_slices()
            return

        print(f">> Reading test snapshot {self.test_data_file}")
        T_all, P_all, Y_all, X_all, Ycoord_all, Z_all = self._read_cfd_snapshot()

        if self.slices:
            print(f">> Selecting cells in {len(self.slices)} slice(s) above T_threshold={self.T_threshold:g} K")
        else:
            print(f">> Sampling {self.n_sample} cells above T_threshold={self.T_threshold:g} K")
        T0, P0, Y0, X0, Ycoord0, Z0, slice_id = self._select_cells(
            T_all, P_all, Y_all, X_all, Ycoord_all, Z_all
        )
        n = T0.shape[0]
        print(f"   >> {n} cells selected")

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
            if self.slices:
                h5f.attrs["slices"] = yaml.dump(self.slices)

            h5f.create_dataset("T_ini", data=T0)
            h5f.create_dataset("P_ini", data=P0)
            h5f.create_dataset("Y_ini", data=Y0)
            h5f.create_dataset("T_cvode", data=T_cvode)
            h5f.create_dataset("Y_cvode", data=Y_cvode)
            h5f.create_dataset("T_ann", data=T_ann)
            h5f.create_dataset("Y_ann", data=Y_ann)
            h5f.create_dataset("cluster", data=cluster_labels)
            h5f.create_dataset("XCEN_X", data=X0)
            h5f.create_dataset("XCEN_Y", data=Ycoord0)
            h5f.create_dataset("XCEN_Z", data=Z0)
            h5f.create_dataset("slice_id", data=slice_id)

        print(">> Done")
