"""
ANN chemistry: advance a batch of states [T, Y_k] by dt with the networks of a model folder.

Shared by the tests of the networks (NNTesting, NN_testing.py) and the stochastic reactors run
with ML inference (ParticlesCloud, ML_inference_flag in dtb_params.yaml), so that both use the
same clustering, transforms and network calls.

The network input is built as in LearningDatabase.process_database (tools/transforms.py); the
processing parameters are read from the learning database (processing_parameters.py).
"""

import os
import sys
from pathlib import Path
import joblib
import pickle
import oyaml as yaml
import h5py

import numpy as np
import cantera as ct
import torch

import ai_reacting_flows.tools.utilities as utils
import ai_reacting_flows.tools.transforms as transforms
from ai_reacting_flows.databases_processing.processing_parameters import read_processing_parameters, MECHANISM_FILE


class NNInference():

    def __init__(self, models_folder, run_folder):
        """Networks, scalers and clustering of models_folder (MODELS/MODEL_<suffix>), trained on the
        learning database given by its networks_params.yaml, relative to run_folder."""

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.models_folder = models_folder
        with open(os.path.join(self.models_folder, "networks_params.yaml"), "r") as file:
            networks_parameters = yaml.safe_load(file)
        self.dataset_path = os.path.join(run_folder, networks_parameters["database_path"])

        # Processing parameters of the learning database (stored in training_data.h5, see processing_parameters.py)
        params = read_processing_parameters(self.dataset_path, run_folder)

        self.dt_var = params["dt_var"]
        self.fuel = params["fuel"] # testing only takes 1 component fuel so far
        self.mech = os.path.join(self.dataset_path, MECHANISM_FILE)   # Mechanism copied in the database folder

        self.log_transform_X = params["log_transform_X"]
        self.log_transform_Y = params["log_transform_Y"]
        # Box-Cox parameter (kept consistent with NN_manager)
        self.lambda_bct = params["lambda_bct"]
        self.threshold = params["threshold"]
        self.remove_N2 = not params["with_N_chemistry"]
        self.output_omegas = params["output_omegas"]
        # General list of species for which input/output log/BCT should be
        # skipped, mirroring LearningDatabase.log_excluded_species.
        self.log_excluded_species = params["log_excluded_species"]

        # Keep track of the clustering feature layout used at database-processing
        # time ("phys", "dt", "all"). This is needed to build a consistent
        # k-means input vector here.
        self.clusterize_on = params["clusterize_on"]

        # CANTERA solution object needed to access species indices
        self.gas = ct.Solution(self.mech)
        utils.check_elements(self.gas)   # Only species made of C, H, O and N are handled
        self.spec_list_ref = self.gas.species_names
        self.nb_species_ref = len(self.spec_list_ref)

        # Getting species and number of species
        self.spec_list_ANN = self.spec_list_ref
        self.nb_species_ANN = self.nb_species_ref
        if self.remove_N2:
            # We drop N2 from the ANN input/output space when with_N_chemistry
            # is disabled in dtb_processing.
            self.spec_list_ANN = [spec for spec in self.spec_list_ref if spec != "N2"]
            self.nb_species_ANN -= 1

        # Find number of clusters -> equal for instance to number of json files in model folder
        self.nb_clusters = len(list(Path(self.models_folder).glob("*.pth")))
        print(f">> {self.nb_clusters} models were found")

        # Check consistency with the processing parameters, could help identify an issue
        assert self.nb_clusters==params["nb_clusters"]

        # If more than one cluster, we need to check the clustering method
        if self.nb_clusters>1:
            if os.path.exists(self.models_folder + "/kmeans_model.pkl"):
                self.clustering_method = "kmeans"
            elif os.path.exists(self.models_folder + "/c_bounds.pkl"):
                self.clustering_method = "progvar"
            else:
                sys.exit("ERROR: There are more than one clusters and no clustering method file in the model folder")
            print(f"   >> Clustering method {self.clustering_method} is used \n")
        else:
            self.clustering_method = None

        # Load models
        self.load_models()

        # Load scalers
        self.load_scalers()

        # Load clustering algorithm if relevant
        if self.nb_clusters>1:
           self.load_clustering_algorithm()

        # Progress variable species (definition of the progress variable used by progvar clustering)
        self.pv_species = params["pv_species"]
        self.pv_ind = [self.gas.species_index(spec) for spec in self.pv_species]

        # Number of atoms (rows: C, H, O, N) in each species and atomic weights [kg/kmol]
        self.atomic_array = utils.get_atomic_array(self.gas)
        self.W_atoms = utils.get_atomic_weights()

#-----------------------------------------------------------------------
#   LOADING
#-----------------------------------------------------------------------

    def load_models(self):
        self.models_list = []

        for i_cluster in range(self.nb_clusters):
            model = torch.load(os.path.join(self.models_folder, f"cluster{i_cluster}_model.pth"), weights_only=False)
            model.to(self.device)
            model.eval()
            # Add to list
            self.models_list.append(model)


    def load_scalers(self):

        self.Xscaler_list = []
        self.Yscaler_list = []

        for i_cluster in range(self.nb_clusters):

            with h5py.File(f"{self.dataset_path:s}/training_data.h5", 'r') as h5file_r:

                grp = h5file_r[f"CLUSTER_{i_cluster}"]

                Xscaler_array = grp['Xscaler'][:]
                Yscaler_array = grp['Yscaler'][:]

                Xscaler_mean = Xscaler_array[:, 0]
                Xscaler_var = Xscaler_array[:, 1]
                #
                Yscaler_mean = Yscaler_array[:, 0]
                Yscaler_var = Yscaler_array[:, 1]

                Xscaler_std = transforms.scaler_std(Xscaler_var)
                Yscaler_std = transforms.scaler_std(Yscaler_var)

                # shape (n_features, 2): column 0 = mean, column 1 = std
                Xscaler_mean_std = np.column_stack((Xscaler_mean, Xscaler_std))
                Yscaler_mean_std = np.column_stack((Yscaler_mean, Yscaler_std))

                self.Xscaler_list.append(Xscaler_mean_std)
                self.Yscaler_list.append(Yscaler_mean_std)


    def load_clustering_algorithm(self):

        if self.clustering_method=="kmeans":
            with open(self.models_folder + "/kmeans_model.pkl", "rb") as f:
                self.kmeans = pickle.load(f)
            self.kmeans_scaler = joblib.load(self.models_folder + "/Xscaler_kmeans.pkl")

        elif self.clustering_method=="progvar":
            with open(self.models_folder + "/c_bounds.pkl", "rb") as input_file:
                self.c_bounds = pickle.load(input_file)

#-----------------------------------------------------------------------
#   CLUSTERING
#-----------------------------------------------------------------------

    def attribute_clusters(self, T, Y, dt, progvar):
        """Clusters (n,) of n states advanced by dt.

        T: temperatures [K] (n), Y: mass fractions [-] (n, n_species) of all species of the mechanism,
        dt: time step [s] (scalar or n), progvar: progress variables [-] (n), used by progvar clustering.
        k-means features built as in LearningDatabase.clusterize_dataset (tools/transforms.py).
        """

        n = len(T)

        if self.clustering_method=="kmeans":

            # Species of the network (N2 removed if necessary)
            Y_ANN = Y[:, [self.spec_list_ref.index(spec) for spec in self.spec_list_ANN]]

            # Clustering on the time step only makes sense with multiple dt
            clusterize_on = self.clusterize_on if self.dt_var else "phys"

            feat = transforms.kmeans_features(T, Y_ANN, np.broadcast_to(dt, n), self.spec_list_ANN,
                                              clusterize_on, self.log_transform_X, self.threshold, self.lambda_bct,
                                              self.log_excluded_species)

            # Scaling with the StandardScaler fitted in LearningDatabase.clusterize_dataset, then k-means
            return self.kmeans.predict(self.kmeans_scaler.transform(feat))

        elif self.clustering_method=="progvar":
            # Cluster i: c_bounds[i] <= progvar < c_bounds[i+1], progvar = 1 in the last cluster
            progvar = np.clip(progvar, 0.0, 1.0)
            clusters = np.searchsorted(self.c_bounds, progvar, side="right") - 1
            return np.clip(clusters, 0, self.nb_clusters - 1)

        # By default if no clustering, we assume that we only have a cluster 0
        else:
            return np.zeros(n, dtype=int)

#-----------------------------------------------------------------------
#   TIME ADVANCEMENT
#-----------------------------------------------------------------------

    def advance(self, T, Y, P, dt, clusters, yk_renormalization):
        """Advance n states by dt with the networks of their clusters, at constant pressure.

        T: temperatures [K] (n), Y: mass fractions [-] (n, n_species) of all species of the mechanism,
        P: pressure [Pa] (scalar or n), dt: time step [s] (scalar or n), clusters: clusters (n)
        from attribute_clusters, yk_renormalization: correct Y to conserve the elements.

        Returns T_new [K] (n), Y_new [-] (n, n_species) and the sum of the predicted mass fractions
        before element correction (n), used to check mass conservation (hybrid ANN/CVODE).
        T_new is deduced from enthalpy conservation, with cp and h_k of the initial state.
        """

        n = len(T)
        dt = np.broadcast_to(dt, n)

        # Species of the network (N2 removed if necessary)
        cols_ANN = [self.spec_list_ref.index(spec) for spec in self.spec_list_ANN]

        # Network input: [T, transformed species, (transformed dt)], as in LearningDatabase.process_database
        # (tools/transforms.py). Species of log_excluded_species stay on a linear scale.
        x_species_t = transforms.transform_species(Y[:, cols_ANN], self.spec_list_ANN, self.log_transform_X,
                                                   self.threshold, self.lambda_bct, self.log_excluded_species)
        log_state = np.column_stack([T, x_species_t])
        if self.dt_var:
            log_state = np.column_stack([log_state, transforms.transform_dt(dt)])

        # Network species at t+dt, one network call per cluster
        Y_ANN_new = np.empty((n, self.nb_species_ANN))
        for i_cluster in np.unique(clusters):
            idx = np.where(clusters == i_cluster)[0]

            # input of NN
            mean_X = self.Xscaler_list[i_cluster][:, 0]
            std_X = self.Xscaler_list[i_cluster][:, 1]
            NN_input = transforms.scale(log_state[idx], mean_X, std_X)

            # Output predicted by ANN
            NN_input = torch.tensor(NN_input, dtype=torch.float64).to(self.device)
            with torch.no_grad():
                NN_output = self.models_list[i_cluster](NN_input)
            NN_output = NN_output.detach().cpu().numpy()

            # Inverse scaling
            mean_Y = self.Yscaler_list[i_cluster][:, 0]
            std_Y = self.Yscaler_list[i_cluster][:, 1]
            y_t = transforms.unscale(NN_output, mean_Y, std_Y)

            # Mass fractions at t+dt (outputs as differences or not, in transformed space or not)
            Y_ANN_new[idx] = transforms.output_to_Yk(x_species_t[idx], y_t, self.spec_list_ANN, self.output_omegas,
                                                     self.log_transform_X, self.log_transform_Y, self.lambda_bct,
                                                     self.log_excluded_species)

        # All species: species not predicted (N2 without N chemistry) are unchanged
        Y_new = Y.copy()
        Y_new[:, cols_ANN] = Y_ANN_new

        # Sum of Yk before element correction (mass conservation check)
        sum_Yk = Y_new.sum(axis=1)

        # Enforcing element conservation
        if yk_renormalization:
            Y_new = self.enforce_elements_balance(Y, Y_new)

        # Deducing T from energy conservation: sum_k h_k Y_k constant, with cp [J/kg/K] and h_k [J/kg] at t
        states = ct.SolutionArray(self.gas, n)
        states.TPY = T, P, Y
        h_k = states.partial_molar_enthalpies / self.gas.molecular_weights
        T_new = T - np.sum(h_k * (Y_new - Y), axis=1) / states.cp_mass

        return T_new, Y_new, sum_Yk


    def enforce_elements_balance(self, Y_old, Y_new):
        """Mass fractions Y_new (n, n_species) corrected to keep the element mass fractions of Y_old:
        C with CO2 (if C is present), H with H2, N with N2, then O with O2."""

        W_k = self.gas.molecular_weights

        # Element mass fractions (n, 4), order C, H, O, N
        def elements(Y):
            return self.W_atoms * ((Y / W_k) @ self.atomic_array.T)

        # Correction of species spec for the element of index i_el
        def correct(Y, spec, i_el, Y_a_in, Y_a_out):
            k = self.gas.species_index(spec)
            Y[:, k] -= (Y_a_out[:, i_el] - Y_a_in[:, i_el]) / ((self.W_atoms[i_el] / W_k[k]) * self.atomic_array[i_el, k])

        Y_new_corr = Y_new.copy()
        Y_a_in = elements(Y_old)
        Y_a_out = elements(Y_new)

        # Carbon: we correct CO2 (only for states containing carbon)
        has_C = Y_a_in[:, 0] > 0
        if has_C.any():
            Y_C = Y_new_corr[has_C]
            correct(Y_C, "CO2", 0, Y_a_in[has_C], Y_a_out[has_C])
            Y_new_corr[has_C] = Y_C

        # Hydrogen: we correct H2
        correct(Y_new_corr, "H2", 1, Y_a_in, Y_a_out)

        # Nitrogen: we correct N2
        correct(Y_new_corr, "N2", 3, Y_a_in, Y_a_out)

        # We perform a new estimation of element mass fractions and then correct O using O2
        Y_a_out = elements(Y_new_corr)
        correct(Y_new_corr, "O2", 2, Y_a_in, Y_a_out)

        return Y_new_corr
