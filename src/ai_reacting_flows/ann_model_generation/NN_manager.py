import os
import shutil
import copy
from datetime import datetime

import numpy as np
import pandas as pd
import oyaml as yaml
import cantera as ct
import matplotlib.pyplot as plt
import h5py

import torch
import torch.nn as nn
import torch.optim as optim

import ai_reacting_flows.tools.utilities as utils
import ai_reacting_flows.tools.transforms as transforms
from ai_reacting_flows.tools.element_projection import project_elements
from ai_reacting_flows.tools.loss_weighting import increment_norm, increment_weights, Omega_decades, error_per_decade, effective_fraction
from ai_reacting_flows.ann_model_generation.NN_models import MLPModel, PerSpeciesMLP, DeepONet, DeepONet_shift
from ai_reacting_flows.databases_processing.processing_parameters import read_processing_parameters, MECHANISM_FILE

torch.set_default_dtype(torch.float64)

activation_functions = {"relu": nn.ReLU, "gelu" : nn.GELU, "tanh" : nn.Tanh, "id" : nn.Identity}
model_type = {"MLP": MLPModel, "PerSpeciesMLP": PerSpeciesMLP, "DeepONet": DeepONet, "DeepONetShift": DeepONet_shift}

class NN_manager():
    def __init__(self, run_folder: str | None = None):

        # Select device for training
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
        # Allow explicit run folder, defaulting to current working directory
        self.run_folder = os.path.abspath(run_folder) if run_folder is not None else os.getcwd()
        
        # Training log: one text file per NN_manager run, placed next to
        # networks_params.yaml in the run folder.
        ts_run = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.run_folder, f"nn_training_{ts_run}.log")

        def _log(msg: str):
            ts = datetime.now().isoformat(timespec="seconds")
            with open(self.log_path, "a") as _f:
                _f.write(f"[{ts}] {msg}\n")

        self._log = _log
        #
        # Networks structures parameters
        #
        with open(os.path.join(self.run_folder, "networks_params.yaml"), "r") as file:
            networks_parameters = yaml.safe_load(file)
        self.dataset_path = os.path.join(self.run_folder, networks_parameters["database_path"])
        
        # Processing parameters of the learning database (stored in training_data.h5, see processing_parameters.py)
        params = read_processing_parameters(self.dataset_path, self.run_folder)

        self.dt_var = params["dt_var"]
        self.rollout = params.get("rollout", False)   # Databases processed before rollout do not have this parameter
        self.threshold = params["threshold"]
        self.remove_N2 = not params["with_N_chemistry"]
        self.log_transform_X = params["log_transform_X"]
        self.log_transform_Y = params["log_transform_Y"]
        self.lambda_bct = params["lambda_bct"]
        self.output_omegas = params["output_omegas"]
        # General list of species for which input/output log/BCT should be
        # skipped. This mirrors LearningDatabase.log_excluded_species.
        self.log_excluded_species = params["log_excluded_species"]

        self.clustering_type = params["clustering_method"]
        self.nb_clusters = params["nb_clusters"]

        # Mechanism copied in the database folder by LearningDatabase
        self.mechanism = os.path.join(self.dataset_path, MECHANISM_FILE)

        # Model folder name
        self.model_name = "MODEL_" + networks_parameters["model_name_suffix"]
        self.new_model_folder = networks_parameters["new_model_folder"]
        # Network(s) structure (1 per cluster)
        self.networks_types = networks_parameters["networks_types"] # list[str] (string in model_type.keys()), 1 per cluster
        self.networks_defs = networks_parameters["networks_def"] # list[str], keys to network parameters in "clusters", 1 per cluster
        self.clusters = networks_parameters["clusters"]

        learning_data = networks_parameters["learning"] # DA: same learning for all clusters ?  (CM: we could change in the future)
        self.optimizer_name = learning_data.get("optimizer", "adam")
        self.loss_name = learning_data.get("loss", "MSE")
        self.initial_learning_rate = learning_data.get("initial_learning_rate", 0.001)
        self.scheduler_option = learning_data.get("scheduler_option", "ExpLR")
        self.LR_decay_rate = learning_data.get("decay_rate", 1.0)
        self.batch_size = learning_data.get("batch_size", 2048)
        self.epochs_list = learning_data["epochs_list"]
        self.val_every = learning_data.get("val_every", 10)
        self.omega_diagnostics_epochs = learning_data.get("omega_diagnostics_epochs", [])   # Epochs of the plots per decade of Omega (also after training)
        self.project_elements = learning_data.get("project_elements", False)              # Element projection in the loss
        self.raw_state_loss_weight = learning_data.get("raw_state_loss_weight", 1.0)      # Weight of the prediction before projection
        self.loss_weighting_p = learning_data.get("loss_weighting_p", 0.0)                # Exponent of the increment weights (0: none)
        self.loss_weighting_eps = learning_data.get("loss_weighting_eps", None)           # Floor of the norm of the scaled increments [-] (units of sigma_k)
        self.loss_weighting_q = learning_data.get("loss_weighting_q", 0.0)                # Exponent of the density weights (0: none)
        self.loss_weighting_min_count = learning_data.get("loss_weighting_min_count", 100) # Floor of the numbers of points per bin of Omega
        self.loss_weighting = self.loss_weighting_p > 0.0 or self.loss_weighting_q > 0.0

        # Increment weights (tools/loss_weighting.py): only for increments of ln Y, with the MSE loss, one step
        if self.loss_weighting:
            if not (self.output_omegas and self.log_transform_Y == 1):
                raise ValueError("Loss weighting (loss_weighting_p or loss_weighting_q > 0) needs output_omegas: true and log_transform_Y: 1 (increments of ln Y)")
            if str(self.loss_name).lower() != "mse":
                raise ValueError("Loss weighting (loss_weighting_p or loss_weighting_q > 0) needs the MSE loss")
            if self.rollout:
                raise ValueError("Loss weighting (loss_weighting_p or loss_weighting_q > 0) is not available with rollout training")
            if self.loss_weighting_eps is None:
                raise ValueError("loss_weighting_eps must be given with loss_weighting_p or loss_weighting_q > 0")


        # Log high-level dataset and processing configuration.
        self._log(
            "INIT "
            f"dt_var={self.dt_var}, "
             f"rollout={self.rollout}, "
            f"project_elements={self.project_elements}, "
             f"remove_N2={self.remove_N2}, "
             f"log_transform_X={self.log_transform_X}, "
             f"log_transform_Y={self.log_transform_Y}, "
             f"lambda_bct={self.lambda_bct}, "
             f"output_omegas={self.output_omegas}, "
             f"log_excluded_species={self.log_excluded_species}, "
            f"loss_weighting_p={self.loss_weighting_p}, "
            f"loss_weighting_eps={self.loss_weighting_eps}, "
            f"loss_weighting_q={self.loss_weighting_q}, "
            f"loss_weighting_min_count={self.loss_weighting_min_count}, "
            f"clustering_type={self.clustering_type}, "
            f"nb_clusters={self.nb_clusters}, "
        )


        # Model's path
        self.directory = f"{self.run_folder:s}/MODELS/{self.model_name:s}"
        if self.new_model_folder:
            print(">> A new folder is created.")
            # Remove folder if already exists
            shutil.rmtree(self.directory, ignore_errors=True)  # DA to CM: raise error / warning ?
            # Create folder
            os.makedirs(self.directory)
        else:
            if not os.path.exists(self.directory):
                raise FileNotFoundError(f"new_model_folder is set to False but model {self.directory} does not exist")
            print(f">> Existing model folder {self.directory} is used. \n")

        # Check if nb_clusters is consistent with the number of groups in training_data.h5
        self._check_nb_clusters()

        # Species of the networks (same across clusters), from the Y columns "<species>_Y"
        # (X columns are "Temperature_X", the same species with "_X", and "dt_X" with multiple dt).
        # Used to apply the per-species inverse log/BCT in conservation diagnostics.
        with h5py.File(f"{self.dataset_path:s}/training_data.h5", 'r') as h5file_r:
            grp0 = h5file_r["CLUSTER_0"]
            self.species_ANN = [str(col)[:-2] for col in grp0['Y_train'].attrs['cols']]

        print("CLUSTERING:")
        print(f">> Number of clusters is: {self.nb_clusters}")
        self._log(f"CLUSTERING nb_clusters={self.nb_clusters}")

        # We copy the mechanism files in order to use them for testing
        if self.new_model_folder:
            shutil.copy(self.mechanism, self.directory)
        
        shutil.copy(os.path.join(self.run_folder, "networks_params.yaml"), self.directory)

        gas = ct.Solution(self.mechanism)
        self.A_element, self.elements = utils.get_molar_mass_atomic_matrix(gas, not self.remove_N2)

        # Thermodynamics of the temperature update of rollout training, as tensors
        if self.rollout:
            self.W_k = torch.tensor(gas.molecular_weights, device=self.device)
            self.nasa7 = [torch.tensor(a, device=self.device) for a in utils.get_nasa7_coefficients(gas)]
            self.cols_ANN = [gas.species_index(spec) for spec in self.species_ANN]
            self.col_N2 = gas.species_index("N2") if self.remove_N2 else None

        # Coefficients of the element projection, as tensors: a_Ai = n_Ai W_A / W_i and c_Ai (n_elements, n_species_ANN)
        if self.project_elements:
            cols = [gas.species_index(spec) for spec in self.species_ANN]
            atomic_array = utils.get_atomic_array(gas)[:, cols]
            a_Ai = utils.get_atomic_weights()[:, None] * atomic_array / gas.molecular_weights[cols]

            # N not considered without N chemistry, as for the conservation diagnostics (order C, H, O, N)
            elements = [0, 1, 2] if self.remove_N2 else [0, 1, 2, 3]

            self.a_Ai = torch.tensor(a_Ai[elements], dtype=torch.float64, device=self.device)
            self.c_Ai = torch.tensor((atomic_array[elements] > 0).astype(float), dtype=torch.float64, device=self.device)

        # Training stats
        if self.new_model_folder:
            os.mkdir(self.directory + "/training")
            os.mkdir(self.directory + "/evaluation" )

            for i_cluster in range(self.nb_clusters):
                os.mkdir(f"{self.directory}/training/cluster_{i_cluster}")
                os.mkdir(f"{self.directory}/evaluation/cluster_{i_cluster}")

            if (self.nb_clusters > 1):
                # Adding copies of clustering parameters for later use in inference
                self.copy_clusterer()

    
    def _check_nb_clusters(self):

        with h5py.File(f"{self.dataset_path:s}/training_data.h5", 'r') as h5file_r:
            top_level_groups = [k for k in h5file_r.keys() if isinstance(h5file_r[k], h5py.Group)]
            nb_clusters_h5 = len(top_level_groups)

            assert self.nb_clusters==nb_clusters_h5
            


    def create_model(self, i_cluster, n_in, n_out):

        network_type = self.networks_types[i_cluster]
        if (network_type not in model_type.keys()):
            raise ValueError(f"network_type '{network_type}' does not exist; valid types: {list(model_type.keys())}")
        else:
            network_parameters = self.clusters[self.networks_defs[i_cluster]]
            if (network_type == "MLP"):
                # Network shapes
                nb_units_in_layers_list = copy.deepcopy(network_parameters["nb_units_in_layers_list"])
                nb_units_in_layers_list.insert(0, n_in)
                nb_units_in_layers_list.append(n_out)
                # Normalize activation names to lowercase to allow e.g. "GeLu"/"GELU"
                layers_activation_list = [activation_functions[str(act).lower()] for act in network_parameters["layers_activation_list"]]
                layers_type = network_parameters["layers_type"]
            
                model = model_type[network_type](self.device, nb_units_in_layers_list, layers_type, layers_activation_list)
            elif (network_type == "PerSpeciesMLP"):
                # Same definition as MLP, for the network of each species: one output instead of n_out
                nb_units_in_layers_list = copy.deepcopy(network_parameters["nb_units_in_layers_list"])
                nb_units_in_layers_list.insert(0, n_in)
                nb_units_in_layers_list.append(1)
                layers_activation_list = [activation_functions[str(act).lower()] for act in network_parameters["layers_activation_list"]]
                layers_type = network_parameters["layers_type"]

                model = model_type[network_type](self.device, nb_units_in_layers_list, layers_type, layers_activation_list, n_out)
            elif (network_type=="DeepONet"):
                # Network shapes
                nb_units_in_layers_list = copy.deepcopy(network_parameters["nb_units_in_layers_list"])
                layers_activation_list = {}
                for k in network_parameters["layers_activation_list"].keys():
                    # Normalize activation names to lowercase to allow e.g. "GeLu"/"GELU"
                    layers_activation_list[k] = [activation_functions[str(act).lower()] for act in network_parameters["layers_activation_list"][k]]
                layers_type = network_parameters["layers_type"]
                n_neurons = network_parameters["n_neurons"]
                nb_units_in_layers_list["branch"].insert(0,n_in)
                nb_units_in_layers_list["branch"].append(n_neurons*n_out)
                nb_units_in_layers_list["trunk"].insert(0,1)
                nb_units_in_layers_list["trunk"].append(n_neurons*n_out)
                model = model_type[network_type](self.device, nb_units_in_layers_list, layers_type, layers_activation_list, n_out, n_neurons)
            elif (network_type=="DeepONetShift"):
                # Network shapes
                nb_units_in_layers_list = copy.deepcopy(network_parameters["nb_units_in_layers_list"])
                layers_activation_list = {}
                for k in network_parameters["layers_activation_list"].keys():
                    # Normalize activation names to lowercase to allow e.g. "GeLu"/"GELU"
                    layers_activation_list[k] = [activation_functions[str(act).lower()] for act in network_parameters["layers_activation_list"][k]]
                layers_type = network_parameters["layers_type"]
                n_neurons = network_parameters["n_neurons"]
                nb_units_in_layers_list["branch"].insert(0,n_in)
                nb_units_in_layers_list["branch"].append(n_neurons*n_out)
                nb_units_in_layers_list["trunk"].insert(0,1)
                nb_units_in_layers_list["trunk"].append(n_neurons*n_out)
                nb_units_in_layers_list["shift"].insert(0,n_in)
                nb_units_in_layers_list["shift"].append(1)
                model = model_type[network_type](self.device, nb_units_in_layers_list, layers_type, layers_activation_list, n_out, n_neurons)
        
        return model
    

    def _physical_Yk(self, X, Y, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std):
        """Mass fractions [-] at t and t+dt from scaled network inputs X and outputs Y.

        Species part of X only (temperature and dt removed). Transforms in tools/transforms.py,
        shared with database processing and inference.
        """
        # Species columns of X (temperature first, time step last when dt_var is used)
        sl = slice(1, -1) if self.dt_var else slice(1, None)

        # Unscaled input (transformed mass fractions at t) and output
        x_species_t = transforms.unscale(X[:, sl], Xscaler_mean[sl], Xscaler_std[sl])
        y_t = transforms.unscale(Y, Yscaler_mean, Yscaler_std)

        Yk_in = transforms.inverse_species(x_species_t, self.species_ANN, self.log_transform_X, self.lambda_bct, self.log_excluded_species)
        Yk_out = transforms.output_to_Yk(x_species_t, y_t, self.species_ANN, self.output_omegas, self.log_transform_X,
                                         self.log_transform_Y, self.lambda_bct, self.log_excluded_species)

        return Yk_in, Yk_out


    def _temperature_after(self, T, Y_old, Y_new, Y_N2):
        """Temperature [K] (n) after a chemistry step, from the conservation of enthalpy at constant pressure:
        same as NNInference.temperature_after, on tensors. T [K] (n), Y_old and Y_new [-] (n, n_species_ANN),
        Y_N2 [-] (n): mass fraction of N2 if it is not predicted (unchanged: only contributes to cp)."""

        h_k, cp_k = utils.species_h_cp(T, self.W_k, *self.nasa7)
        cp = torch.sum(cp_k[:, self.cols_ANN] * Y_old, dim=1)

        # N2 not predicted
        if self.remove_N2:
            cp = cp + cp_k[:, self.col_N2] * Y_N2

        return T - torch.sum(h_k[:, self.cols_ANN] * (Y_new - Y_old), dim=1) / cp


    def _projected_output(self, X, Y_old, Y_new, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std):
        """Mass fractions Y_new [-] (n, n_species_ANN) projected on the elements of Y_old [-] (as yk_renormalization: 3
        at inference), and the scaled network output that would give them, to compare with the scaled targets."""

        # Species columns of X (temperature first, time step last when dt_var is used)
        sl = slice(1, -1) if self.dt_var else slice(1, None)
        x_species_t = transforms.unscale(X[:, sl], Xscaler_mean[sl], Xscaler_std[sl])

        Y_projected = project_elements(Y_old, Y_new, self.a_Ai, self.c_Ai)
        y_t = transforms.Yk_to_output(x_species_t, Y_projected, self.species_ANN, self.output_omegas, self.log_transform_X,
                                      self.log_transform_Y, self.threshold, self.lambda_bct, self.log_excluded_species)

        return Y_projected, transforms.scale(y_t, Yscaler_mean, Yscaler_std)


    def _loss(self, loss_fn, y_pred, y_target, weights=None):
        """loss_fn on scaled outputs, with optional weights (n, 1) of the MSE loss (tools/loss_weighting.py):
        mean of w (y_pred - y_target)^2, computed as the MSE of sqrt(w) y_pred and sqrt(w) y_target."""

        if weights is None:
            return loss_fn(y_pred, y_target)

        sqrt_w = torch.sqrt(weights)
        return loss_fn(sqrt_w * y_pred, sqrt_w * y_target)


    def _one_step_loss(self, loss_fn, X, y_pred, y_target, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std, raw_weight=0.0, weights=None):
        """Loss of one step, from scaled network outputs y_pred and targets y_target. With project_elements: loss on
        the projected state, plus raw_weight times the loss on the prediction before projection (see docs/theory.md).
        weights: loss weights (n, 1) of loss_weighting_p and loss_weighting_q, None for the usual loss."""

        if self.project_elements:
            Y_in, Y_new = self._physical_Yk(X, y_pred, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std)
            _, y_projected = self._projected_output(X, Y_in, Y_new, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std)

            return self._loss(loss_fn, y_projected, y_target, weights) + raw_weight * self._loss(loss_fn, y_pred, y_target, weights)

        # Usual loss
        else:
            return self._loss(loss_fn, y_pred, y_target, weights)


    def _loss_weights(self, Y, Yscaler_mean, Yscaler_std, ref=None):
        """Increment weights (n, 1) of the points of scaled outputs Y (numpy array), the reference of the training
        points (bins of Omega and normalization) and the norms Omega (n) of their scaled increments (see
        tools/loss_weighting.py). ref: reference of the training points, None to compute it."""

        # Increments of ln Y_k, scaled by their standard deviations (mean not removed)
        omega = transforms.unscale(Y, Yscaler_mean, Yscaler_std)
        Omega = increment_norm(omega, Yscaler_std)

        w, ref = increment_weights(Omega, self.loss_weighting_p, self.loss_weighting_eps,
                                   self.loss_weighting_q, self.loss_weighting_min_count, ref)

        return w[:, None], ref, Omega


    def _rollout_loss(self, model, loss_fn, X, Y_rollout, Y_N2, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std, raw_weight=0.0):
        """Loss of a rollout: mean over the steps of _rollout_step_losses. With one step, it is the usual loss."""

        return self._rollout_step_losses(model, loss_fn, X, Y_rollout, Y_N2, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std, raw_weight).mean()


    def _rollout_step_losses(self, model, loss_fn, X, Y_rollout, Y_N2, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std, raw_weight=0.0):
        """Loss of each step of a rollout, tensor (n_steps): from the states of X, the network is applied n_steps
        times, each time on its own prediction, with temperatures from enthalpy conservation. Loss on the error of
        the predicted state, in units of the standard deviation of the outputs (see docs/theory.md). All steps use
        the network of the cluster of the initial state. With project_elements, the chain carries the projected state,
        and each step has the loss of _one_step_loss.

        X: scaled inputs at t (n, n_in), Y_rollout: transformed (not scaled) mass fractions at t+k*dt
        (n, n_steps, n_species_ANN), Y_N2: mass fractions of N2 [-] (n), not used if N2 is predicted.
        """
        n_steps = Y_rollout.shape[1]

        losses = []
        for k in range(n_steps):

            y_pred = model(X)

            # Current state, as network input: transformed and physical (clipped). Physical state of the next step
            x_species_t = transforms.unscale(X[:, 1:], Xscaler_mean[1:], Xscaler_std[1:])
            Y_in, Y_new = self._physical_Yk(X, y_pred, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std)

            # Output that gives the true state at t+k*dt. Differences are taken from the PREDICTED state,
            # with its gradient: the loss is on the state error, also with output_omegas
            if self.output_omegas:

                # Differences in transformed space
                if self.log_transform_Y > 0:
                    target = Y_rollout[:, k] - x_species_t

                # Differences of mass fractions
                else:
                    target = Y_rollout[:, k] - Y_in

            # Outputs are the state at t+k*dt
            else:
                target = Y_rollout[:, k]

            y_target = transforms.scale(target, Yscaler_mean, Yscaler_std)

            # Physical state before this step, for the element projection and the temperature update.
            # Not Y_in after the first step: clipping would add spurious composition changes to the enthalpy budget
            Y_old = Y_in if k == 0 else Y_previous

            if self.project_elements:
                Y_new, y_projected = self._projected_output(X, Y_old, Y_new, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std)
                losses.append(loss_fn(y_projected, y_target) + raw_weight * loss_fn(y_pred, y_target))

            # Usual loss
            else:
                losses.append(loss_fn(y_pred, y_target))

            # Next input, built as at inference (NNInference.advance): the temperature update uses the predicted
            # state, not clipped (the clip only applies to the network input)
            if k < n_steps - 1:
                T = transforms.unscale(X[:, 0], Xscaler_mean[0], Xscaler_std[0])
                T_new = self._temperature_after(T, Y_old, Y_new, Y_N2)
                Y_previous = Y_new
                x_species_new = transforms.transform_species(Y_new, self.species_ANN, self.log_transform_X, self.threshold,
                                                             self.lambda_bct, self.log_excluded_species)
                X = transforms.scale(torch.column_stack([T_new, x_species_new]), Xscaler_mean, Xscaler_std)

        return torch.stack(losses)


    def _log_error_vs_Omega(self, i_cluster, model, X_val, Y_val, Yscaler_mean, Yscaler_std, W_val, tag):
        """Log and plot the errors of the validation points per decade of Omega (see tools/loss_weighting.py), for
        outputs that are increments. Error of the network output, in units of the standard deviations of the outputs.
        W_val: loss weights (n, 1) of the validation points, None for the usual loss. tag: epoch of the model, in the
        log lines and the names of the plots (e.g. epoch00100, or best_epoch00480 for the model kept after training)."""

        model.eval()
        with torch.no_grad():
            err = torch.linalg.norm(model(X_val) - Y_val, dim=1).cpu().numpy()
        model.train()

        std = Yscaler_std.cpu().numpy()
        omega = transforms.unscale(Y_val.cpu().numpy(), Yscaler_mean.cpu().numpy(), std)
        w = None if W_val is None else W_val[:, 0].cpu().numpy()

        Omega = increment_norm(omega, std)
        rows = error_per_decade(Omega, err, w)
        norm_share, projection_share = self._gradient_shares(model, X_val, Y_val, W_val, Omega_decades(Omega))

        msg = f"ERROR_VS_OMEGA cluster={i_cluster}, {tag}, median_error={np.median(err):.3e}"
        print(f"    >> {msg}")
        self._log(msg)

        for (decade, n_points, median_err, median_ratio, loss_share, e_decade), g_norm, g_projection in zip(rows, norm_share, projection_share):
            msg = (f"ERROR_VS_OMEGA cluster={i_cluster}, {tag}, log10_Omega={decade}, n={n_points}, "
                   f"median_error={median_err:.3e}, median_error_over_Omega={median_ratio:.3e}, e_decade={e_decade:.3e}, "
                   f"loss_share={loss_share:.3f}, "
                   f"gradient_norm_share={g_norm:.3f}, gradient_projection_share={g_projection:.3f}")
            print(f"    >> {msg}")
            self._log(msg)

        self._plot_error_vs_Omega(i_cluster, rows, norm_share, projection_share, tag)


    def _gradient_shares(self, model, X_val, Y_val, W_val, decades):
        """Shares of the decades of Omega in the gradient of the loss with respect to the network parameters theta,
        on the points X_val, Y_val (scaled), with weights W_val (n, 1) (None: weights 1). decades (n): decade of
        each point (Omega_decades). Loss of the network output, without the element projection.

        For each decade d, g_d = dL_d/dtheta, with L_d = sum over the points of d of w_i ||r_i||^2 / (n K), and
        g = sum_d g_d the gradient of the whole loss. Returns, per decade (sorted as np.unique(decades)):
            - norm share ||g_d|| / sum_d ||g_d||: how strongly the decade pushes (includes the cancellations between
              its points, not between decades);
            - projection share (g_d . g) / ||g||^2: contribution of the decade to the gradient, summing to 1, negative
              if the decade pulls against the others.
        """

        params = [param for param in model.parameters() if param.requires_grad]

        # Weighted squared error of each point, with the graph for the gradients
        squared_error = torch.sum((model(X_val) - Y_val)**2, dim=1)
        if W_val is not None:
            squared_error = squared_error * W_val[:, 0]
        normalization = Y_val.shape[0] * Y_val.shape[1]

        # Gradient of the loss of each decade, flattened over all parameters (n_decades, n_parameters)
        decades = torch.as_tensor(decades, device=squared_error.device)
        gradients = []
        for decade in torch.unique(decades):
            loss_decade = squared_error[decades == decade].sum() / normalization
            grads = torch.autograd.grad(loss_decade, params, retain_graph=True)
            gradients.append(torch.cat([grad.flatten() for grad in grads]))
        G = torch.stack(gradients)

        g = G.sum(dim=0)
        norms = torch.linalg.norm(G, dim=1)

        norm_share = (norms / norms.sum()).detach().cpu().numpy()
        projection_share = ((G @ g) / (g @ g)).detach().cpu().numpy()

        return norm_share, projection_share


    def _plot_error_vs_Omega(self, i_cluster, rows, norm_share, projection_share, tag):
        """Plots per decade of Omega (rows of error_per_decade, gradient shares of _gradient_shares), in
        training/cluster_<i>/, with the tag (epoch) at the end of the names: numbers of points (Omega_points_<tag>.png),
        contributions of one point of the decade to the loss (Omega_loss_per_point_<tag>.png) and to the gradient
        (Omega_gradient_per_point_<tag>.png), median relative error and relative error of the decade e_decade
        (Omega_relative_error_<tag>.png). Contributions per point are relative to the average point (1): share of the
        decade divided by its fraction of points N_d / N. Training is badly balanced if points of decades with a median
        relative error well above 1 contribute much to the gradient."""

        decades, n_points, _, median_ratio, loss_share, e_decade = map(np.array, zip(*rows))
        x = np.arange(len(decades))     # One position per decade present (no gap for Omega = 0, at the floor)
        folder = f"{self.directory}/training/cluster_{i_cluster}"

        # Fraction of the points in each decade, to turn shares into contributions per point
        point_fraction = n_points / n_points.sum()

        # Numbers of points
        fig, ax = plt.subplots()
        ax.bar(x, n_points, color="grey")
        ax.set_yscale("log")
        ax.set_ylabel("Validation points")
        ax.set_xticks(x, decades)
        ax.set_xlabel(r"Decade of $\log_{10}\Omega$")
        fig.tight_layout()
        fig.savefig(f"{folder}/Omega_points_{tag}.png")
        plt.close(fig)

        # Loss of one point of the decade, relative to the average point
        fig, ax = plt.subplots()
        ax.bar(x, loss_share / point_fraction, color="k")
        ax.axhline(1.0, ls="--", color="grey")
        ax.set_yscale("log")
        ax.set_ylabel(r"Loss per point $w\,\|r\|^2$ / average point [-]")
        ax.set_xticks(x, decades)
        ax.set_xlabel(r"Decade of $\log_{10}\Omega$")
        fig.tight_layout()
        fig.savefig(f"{folder}/Omega_loss_per_point_{tag}.png")
        plt.close(fig)

        # Gradient of one point of the decade, relative to the average point (projection can be negative)
        fig, ax = plt.subplots()
        ax.bar(x - 0.2, norm_share / point_fraction, width=0.4, color="b", label=r"Norm: $\|g_d\| / \sum \|g_d\|$")
        ax.bar(x + 0.2, projection_share / point_fraction, width=0.4, color="g", label=r"Projection: $g_d \cdot g / \|g\|^2$")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.axhline(1.0, ls="--", color="grey")
        ax.set_yscale("symlog", linthresh=1.0e-2)
        ax.set_ylabel("Gradient per point / average point [-]")
        ax.legend()
        ax.set_xticks(x, decades)
        ax.set_xlabel(r"Decade of $\log_{10}\Omega$")
        fig.tight_layout()
        fig.savefig(f"{folder}/Omega_gradient_per_point_{tag}.png")
        plt.close(fig)

        # Accuracy of each decade
        fig, ax = plt.subplots()
        ax.plot(x, median_ratio, color="b", marker="o", label=r"Median of $\|r\|\,/\,\Omega$")
        ax.plot(x, e_decade, color="r", marker="s", label=r"$e_{decade} = \sqrt{\sum \|r\|^2 / \sum \Omega^2}$")
        ax.axhline(1.0, ls="--", color="k")
        ax.set_yscale("log")
        ax.set_ylabel("Relative error [-]")
        ax.legend()
        ax.set_xticks(x, decades)
        ax.set_xlabel(r"Decade of $\log_{10}\Omega$")
        fig.tight_layout()
        fig.savefig(f"{folder}/Omega_relative_error_{tag}.png")
        plt.close(fig)


    def train_model(self, i_cluster, model, loss_fn, optimizer, scheduler, X_train, X_val, Y_train, Y_val, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std):

        # Increment weights of the loss (None: usual loss), validation normalized as training
        W_train, W_val = None, None
        if self.loss_weighting:
            W_train, ref, Omega = self._loss_weights(Y_train, Yscaler_mean, Yscaler_std)
            W_val, _, _ = self._loss_weights(Y_val, Yscaler_mean, Yscaler_std, ref)

            msg = (f"LOSS_WEIGHTS cluster={i_cluster}, min={W_train.min():.3e}, max={W_train.max():.3e}, "
                   f"effective_fraction={effective_fraction(W_train):.3f}, "
                   f"bins of log10(Omega)={len(ref['counts'])} (min/max points: {ref['counts'][ref['counts'] > 0].min()}/{ref['counts'].max()}), "
                   f"Omega quantiles 1%/50%/99%={np.quantile(Omega, 0.01):.3e}/{np.median(Omega):.3e}/{np.quantile(Omega, 0.99):.3e}")
            print(f"    >> {msg}")
            self._log(msg)

            W_train = torch.tensor(W_train, dtype=torch.float64, device=self.device)
            W_val = torch.tensor(W_val, dtype=torch.float64, device=self.device)

        X_train = torch.tensor(X_train, dtype=torch.float64)
        Y_train = torch.tensor(Y_train, dtype=torch.float64)
        X_val = torch.tensor(X_val, dtype=torch.float64)
        Y_val = torch.tensor(Y_val, dtype=torch.float64)
        A_element = torch.tensor(self.A_element, dtype=torch.float64)
        Xscaler_mean = torch.from_numpy(Xscaler_mean).to(torch.float64)
        Xscaler_std = torch.from_numpy(Xscaler_std).to(torch.float64)
        Yscaler_mean = torch.from_numpy(Yscaler_mean).to(torch.float64)
        Yscaler_std = torch.from_numpy(Yscaler_std).to(torch.float64)

        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)
        X_val = X_val.to(self.device)
        Y_val = Y_val.to(self.device)

        # Rollout targets, and N2 for the temperature update
        if self.rollout:
            Y_rollout_train, Y_rollout_val, Y_N2_train, Y_N2_val = [torch.tensor(data, dtype=torch.float64, device=self.device)
                                                                    for data in self.read_rollout_data(i_cluster)]

        A_element = A_element.to(self.device)    # Array to store the loss and validation loss

        Xscaler_mean = Xscaler_mean.to(self.device)
        Xscaler_std = Xscaler_std.to(self.device)
        #
        Yscaler_mean = Yscaler_mean.to(self.device)
        Yscaler_std = Yscaler_std.to(self.device)

        n_epochs = self.epochs_list[i_cluster]

        loss_list = np.empty(n_epochs)

        # Validation / conservation computed every "val_every" epochs (default 10)
        val_every = int(self.val_every)

        epochs = np.arange(n_epochs)
        epochs_small = epochs[::val_every]
        n_val_points = len(epochs_small)

        # Array to store sum of mass fractions: mean, min and max
        stats_sum_yk = np.empty((n_val_points, 3))

        # Array to store elements conservation: mean, min and max
        stats_A_elements = np.empty((n_val_points, A_element.shape[0], 3))

        best_val_loss = float("inf")
        best_epoch = n_epochs - 1   # Epoch of the model kept after training (last one if no validation)
        best_state_dict = None

        val_loss_list = np.empty(n_val_points)

        # Log basic training configuration for this cluster
        self._log(
            "TRAIN_START "
            f"cluster={i_cluster}, "
            f"n_epochs={n_epochs}, "
            f"batch_size={self.batch_size}, "
            f"optimizer={self.optimizer_name}, "
            f"loss={self.loss_name}, "
            f"initial_lr={self.initial_learning_rate}, "
            f"scheduler_option={self.scheduler_option}"
        )

        for epoch in range(n_epochs):

            # Shuffling training data before each epoch
            perm = torch.randperm(len(X_train), device=self.device)
            X_train, Y_train = X_train[perm], Y_train[perm]
            if W_train is not None:
                W_train = W_train[perm]
            if self.rollout:
                Y_rollout_train, Y_N2_train = Y_rollout_train[perm], Y_N2_train[perm]

            # Training parameters
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, len(X_train), self.batch_size):

                Xbatch = X_train[i:i+self.batch_size]

                # Network applied on its own predictions
                if self.rollout:
                    loss = self._rollout_loss(model, loss_fn, Xbatch, Y_rollout_train[i:i+self.batch_size], Y_N2_train[i:i+self.batch_size],
                                              Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std, self.raw_state_loss_weight)

                # One step
                else:
                    y_pred = model(Xbatch)
                    ybatch = Y_train[i:i+self.batch_size]
                    wbatch = None if W_train is None else W_train[i:i+self.batch_size]
                    loss = self._one_step_loss(loss_fn, Xbatch, y_pred, ybatch, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std,
                                               self.raw_state_loss_weight, wbatch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
                # print("Targets stats:", Y_train.min().item(), Y_train.max().item())
                # print("Preds stats:", y_pred.min().item(), y_pred.max().item())
                # input("Press Enter to continue.")

            loss_list[epoch] = epoch_loss / max(1, n_batches)

            # Track learning rate evolution
            before_lr = optimizer.param_groups[0]["lr"]

            # Computing validation loss and mass conservation metric (only every "val_every" epochs as it is expensive).
            # For ReduceLROnPlateau we want to drive the scheduler from the validation loss.
            val_loss = None
            if epoch % val_every == 0:
                model.eval()  # evaluation mode
                with torch.no_grad():

                    # VALIDATION LOSS (with project_elements: on the projected state only, used at inference)
                    y_val_pred = model(X_val)
                    if self.rollout:
                        val_loss = self._rollout_loss(model, loss_fn, X_val, Y_rollout_val, Y_N2_val, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std)
                    else:
                        val_loss = self._one_step_loss(loss_fn, X_val, y_val_pred, Y_val, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std,
                                                       weights=W_val)

                    # Unweighted validation loss, to compare models trained with different weights
                    val_loss_unweighted = None
                    if W_val is not None:
                        val_loss_unweighted = self._one_step_loss(loss_fn, X_val, y_val_pred, Y_val, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std)

                    # Physical mass fractions at t (input) and t+dt (prediction)
                    yval_in, yk_abs = self._physical_Yk(X_val, y_val_pred, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std)

                    # SUM OF MASS FRACTION
                    sum_yk = yk_abs.sum(axis=1)
                    sum_yk = sum_yk.detach().cpu().numpy()
                    idx = epoch // val_every
                    stats_sum_yk[idx,0] = sum_yk.mean() 
                    stats_sum_yk[idx,1] = sum_yk.min()
                    stats_sum_yk[idx,2] = sum_yk.max()

                    # ELEMENTS CONSERVATION
                    ye_in = torch.matmul(A_element, torch.transpose(yval_in, 0, 1))
                    ye_out = torch.matmul(A_element, torch.transpose(yk_abs, 0, 1))
                    delta_ye = (ye_out - ye_in)/(ye_in+1e-10)
                    delta_ye = delta_ye.detach().cpu().numpy()
                    stats_A_elements[idx, :, 0] = delta_ye.mean(axis=1)
                    stats_A_elements[idx, :, 1] = delta_ye.min(axis=1)
                    stats_A_elements[idx, :, 2] = delta_ye.max(axis=1)

                model.train()   # Back to training mode

                if best_state_dict is None or val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_state_dict = copy.deepcopy(model.state_dict())
                    best_epoch = epoch

                val_loss_list[idx] = val_loss.item()

            # Scheduler step: use validation loss for ReduceLROnPlateau when available;
            # for other schedulers, step once per epoch.
            if scheduler is not None:
                if self.scheduler_option == "ReduceLROnPlateau" and val_loss is not None:
                    scheduler.step(val_loss.item())
                elif self.scheduler_option != "ReduceLROnPlateau":
                    scheduler.step()

            after_lr = optimizer.param_groups[0]["lr"]

            msg = (
                f"EPOCH cluster={i_cluster} epoch={epoch} "
                f"lr={before_lr:.6e}->{after_lr:.6e} "
                f"train_loss={loss_list[epoch]:.6e}"
            )
            if epoch % val_every == 0 and val_loss is not None:
                msg += f" val_loss={val_loss.item():.6e}"
                if val_loss_unweighted is not None:
                    msg += f" val_loss_unweighted={val_loss_unweighted.item():.6e}"
            self._log(msg)

            print(f"Finished epoch {epoch}")
            print(f"    >> lr: {before_lr} -> {after_lr}")
            print(f"    >> Loss (mean over batches): {loss_list[epoch]}")
            if epoch % val_every == 0 and val_loss is not None:
                print(f"    >> Validation loss: {val_loss.item()}")

            # Errors, loss and gradient per decade of Omega at the selected epochs
            if self.output_omegas and epoch in self.omega_diagnostics_epochs:
                self._log_error_vs_Omega(i_cluster, model, X_val, Y_val, Yscaler_mean, Yscaler_std, W_val, f"epoch{epoch:05d}")

        # Restore best model (based on validation loss) before returning
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # Postprocessing: error of the best model versus the size of the increments
        if self.output_omegas:
            self._log_error_vs_Omega(i_cluster, model, X_val, Y_val, Yscaler_mean, Yscaler_std, W_val, f"best_epoch{best_epoch:05d}")

        # Postprocessing: validation loss of each step of the rollout, with the best model
        if self.rollout:
            model.eval()
            with torch.no_grad():
                val_step_losses = self._rollout_step_losses(model, loss_fn, X_val, Y_rollout_val, Y_N2_val, Xscaler_mean, Xscaler_std, Yscaler_mean, Yscaler_std)
            model.train()

            val_step_losses = val_step_losses.cpu().numpy()
            pd.DataFrame({"step": np.arange(1, len(val_step_losses) + 1), "val_loss": val_step_losses}).to_csv(
                f"{self.directory}/training/cluster_{i_cluster}/rollout_val_loss_per_step.csv", index=False)

            self._log(f"ROLLOUT_VAL_LOSS_PER_STEP cluster={i_cluster}, val_loss={val_step_losses.tolist()}")

        self._log(
            "TRAIN_END "
            f"cluster={i_cluster}, "
            f"best_val_loss={best_val_loss:.6e}"
        )

        return epochs, epochs_small, loss_list, val_loss_list, stats_sum_yk, stats_A_elements
    

    def train_all_clusters(self):

        for i_cluster in range(self.nb_clusters):

            # Reading training and validation data
            X_train, X_val, Y_train, Y_val = self.read_training_data(i_cluster)

            # Read scalers
            Xscaler_mean, Xscaler_var, Yscaler_mean, Yscaler_var = self.get_scalers_stats(i_cluster)

            # Create model
            model = self.create_model(i_cluster, X_val.shape[1], Y_val.shape[1])

            # Optimizer and loss function
            opt_name = str(self.optimizer_name).lower()
            if opt_name == "adam":
                optimizer = optim.Adam(model.parameters(), lr=self.initial_learning_rate)
            elif opt_name == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=self.initial_learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer '{opt_name}'")

            loss_name = str(self.loss_name).lower()
            if loss_name == "mse":
                loss_fn = nn.MSELoss()
            elif loss_name == "l1":
                loss_fn = nn.L1Loss()
            elif loss_name == "huber":
                loss_fn = nn.HuberLoss()
            else:
                raise ValueError(f"Unsupported loss '{loss_name}'")

            # Set scheduler
            if self.scheduler_option == "ExpLR":
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.LR_decay_rate)
            elif self.scheduler_option == "ReduceLROnPlateau":
                # Plateau scheduler is driven by validation loss in train_model, so it is only stepped
                # every val_every epochs: patience is a number of validations (patience * val_every epochs).
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=0.5,
                    patience=10,
                    mode="min",
                )
            elif self.scheduler_option == "None" or self.scheduler_option is None:
                # No scheduler: pass None, train_model will skip scheduler.step
                scheduler = None
            else:
                raise ValueError("Only scheduler implemented yet is ExpLR, ReduceLROnPlateau or None")
            
            # Perform training
            epochs, epochs_small, loss_list, val_loss_list, stats_sum_yk, stats_A_elements = self.train_model(i_cluster, model, loss_fn, optimizer, scheduler, X_train, X_val, Y_train, Y_val, Xscaler_mean, transforms.scaler_std(Xscaler_var), Yscaler_mean, transforms.scaler_std(Yscaler_var))

            # Plot training monitoring data (loss, conservation,etc...)
            self.plot_losses_conservation(i_cluster, epochs, epochs_small, loss_list, val_loss_list, stats_sum_yk, stats_A_elements)

            # Save model (torch format and custom h5 format)
            torch.save(model, os.path.join(self.directory, f"cluster{i_cluster}_model.pth"))
            self.save_pt_model_to_h5(model, os.path.join(self.directory, f"cluster{i_cluster}_model.h5"))

            # Save normalization parameters
            np.savetxt(os.path.join(self.directory, f"norm_param_X_cluster{i_cluster}.dat"), np.vstack([Xscaler_mean, Xscaler_var]).T)
            np.savetxt(os.path.join(self.directory, f"norm_param_Y_cluster{i_cluster}.dat"), np.vstack([Yscaler_mean, Yscaler_var]).T)


    def plot_losses_conservation(self, i_cluster, epochs, epochs_small, loss_list, val_loss_list, stats_sum_yk, stats_A_elements):
        # LOSSES
        fig, ax = plt.subplots()

        ax.plot(epochs, loss_list, color="k", label="Training")
        ax.plot(epochs_small, val_loss_list, color="r", label = "Validation")

        ax.set_yscale('log')

        ax.legend()

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        plt.savefig( f"{self.directory}/training/cluster_{i_cluster}/loss.png")

        # MASS CONSERVATION
        fig, ax = plt.subplots()

        ax.plot(epochs_small, stats_sum_yk[:,0], color="k")
        ax.plot(epochs_small, stats_sum_yk[:,1], color="k", ls="--")
        ax.plot(epochs_small, stats_sum_yk[:,2], color="k", ls="--")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\sum_k \ Y_k$")

        plt.savefig( f"{self.directory}/training/cluster_{i_cluster}/mass_conservation.png")

        # ELEMENTS CONSERVATION

        # Elements in the same order as the rows of A_element (and of stats_A_elements)
        n = len(self.elements)
        ncols = 2
        nrows = -(-n // ncols)  # ceil division

        fig, axes = plt.subplots(nrows, ncols, squeeze=False)
        axes = axes.flat

        for ax, (i, label) in zip(axes, enumerate(self.elements)):
            ax.plot(epochs_small, 100*stats_A_elements[:, i, 0], color="k")
            ax.plot(epochs_small, 100*stats_A_elements[:, i, 1], color="k", ls="--")
            ax.plot(epochs_small, 100*stats_A_elements[:, i, 2], color="k", ls="--")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(rf"$\Delta Y_{{{label}}}$ $(\%)$")

        # hide any unused axes (e.g. odd number of elements)
        for ax in list(axes)[n:]:
            ax.set_visible(False)

        fig.tight_layout()


        plt.savefig( f"{self.directory}/training/cluster_{i_cluster}/elements_conservation.png")

    def copy_clusterer(self):
        if self.clustering_type=="progvar":
            shutil.copy(self.dataset_path + "/c_bounds.pkl", self.directory)
        elif self.clustering_type=="kmeans":
            shutil.copy(self.dataset_path + "/kmeans_model.pkl", self.directory)
            shutil.copy(self.dataset_path + "/Xscaler_kmeans.pkl", self.directory)
            shutil.copy(self.dataset_path + "/kmeans_norm.dat", self.directory)
            shutil.copy(self.dataset_path + "/km_centroids.dat", self.directory)
        else:
            if self.nb_clusters > 1:
                raise ValueError("nb_cluster > 1 but cluster_type undefined (should be 'progvar' or 'kmeans')")

    def read_training_data(self, i_cluster):

        with h5py.File(f"{self.dataset_path:s}/training_data.h5", 'r') as h5file_r:

            grp = h5file_r[f"CLUSTER_{i_cluster}"]

            X_train = grp['X_train'][:]
            Y_train = grp['Y_train'][:]
            X_val   = grp['X_val'][:]
            Y_val   = grp['Y_val'][:]

        return X_train, X_val, Y_train, Y_val


    def read_rollout_data(self, i_cluster):

        with h5py.File(f"{self.dataset_path:s}/training_data.h5", 'r') as h5file_r:

            grp = h5file_r[f"CLUSTER_{i_cluster}"]

            return grp['Y_rollout_train'][:], grp['Y_rollout_val'][:], grp['Y_N2_train'][:], grp['Y_N2_val'][:]

    def get_scalers_stats(self, i_cluster):

        with h5py.File(f"{self.dataset_path:s}/training_data.h5", 'r') as h5file_r:
            
            grp = h5file_r[f"CLUSTER_{i_cluster}"]

            Xscaler_array = grp['Xscaler'][:]
            Yscaler_array = grp['Yscaler'][:]

            Xscaler_mean = Xscaler_array[:,0]
            Xscaler_var = Xscaler_array[:,1]
            #
            Yscaler_mean = Yscaler_array[:,0]
            Yscaler_var = Yscaler_array[:,1]


        return Xscaler_mean, Xscaler_var, Yscaler_mean, Yscaler_var


    def save_pt_model_to_h5(self, model, h5_path):
        """Export PyTorch model parameters to HDF5 format.

        Layout is designed for cross-framework compatibility (e.g., TensorFlow),
        using 'kernel:0' and 'bias:0' datasets with weights transposed to match
        expected dense layer formats. A PerSpeciesMLP is written as one group
        species_<i> per output i, each with the layers of its MLP.
        """
        model.eval()

        def save_module(group, module, module_name, parent_activation_map):
            """
            Recursively save modules with parameters only, skipping pure activations.
            """
            # Skip pure activations (no params, no children)
            if len(list(module.parameters(recurse=False))) == 0 and len(list(module.children())) == 0:
                return

            # Create a group for this module
            layer_group = group.create_group(module_name)

            # Save activation if exists
            activation_name = parent_activation_map.get(module_name, 'none')
            layer_group.attrs['activation'] = activation_name

            # Save parameters of this module (Linear, etc.)
            for param_name, param in module.named_parameters(recurse=False):
                data = param.detach().cpu().numpy()
                if param_name == "weight":
                    layer_group.create_dataset("kernel:0", data=data.T)   # We noticed that transpose of matrix need to be considered (differences in pytorch vs tf layouts)
                elif param_name == "bias":
                    layer_group.create_dataset("bias:0", data=data)

            # Recursively save children (ResidualBlock submodules)
            for child_name, child_module in module.named_children():
                save_module(layer_group, child_module, child_name, parent_activation_map)

        with h5py.File(h5_path, "w") as f:
            # One MLP per species: one group per species, in the order of the outputs
            if isinstance(model, PerSpeciesMLP):
                for i_species in range(model.n_species):
                    species_group = f.create_group(f"species_{i_species}")
                    for name, activation, kernel, bias in model.species_layers(i_species):
                        layer_group = species_group.create_group(name)
                        layer_group.attrs["activation"] = activation
                        layer_group.create_dataset("kernel:0", data=kernel)
                        layer_group.create_dataset("bias:0", data=bias)
            else:
                # Save each top-level layer directly, no 'model' group
                for layer_name, module in model.model.named_children():  # note: model.model is nn.Sequential
                    save_module(f, module, layer_name, getattr(model, 'activation_map', {}))
