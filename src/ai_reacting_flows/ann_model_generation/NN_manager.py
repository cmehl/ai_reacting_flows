import sys
import os
import shutil
import joblib
import copy

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
from ai_reacting_flows.ann_model_generation.NN_models import MLPModel, DeepONet, DeepONet_shift

torch.set_default_dtype(torch.float64)

activation_functions = {"ReLU": nn.ReLU, "GeLU" : nn.GELU, "tanh" : nn.Tanh, "Id" : nn.Identity}
model_type = {"MLP": MLPModel, "DeepONet": DeepONet, "DeepONetShift": DeepONet_shift}

class NN_manager():
    def __init__(self, run_folder: str | None = None):

        # Select device for training
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
        # Allow explicit run folder, defaulting to current working directory
        self.run_folder = os.path.abspath(run_folder) if run_folder is not None else os.getcwd()
        #
        # Networks structures parameters
        #
        with open(os.path.join(self.run_folder, "networks_params.yaml"), "r") as file:
            networks_parameters = yaml.safe_load(file)
        self.dataset_path = os.path.join(self.run_folder, networks_parameters["database_path"])
        
        with open(f"{self.dataset_path}/dtb_processing.yaml", "r") as file:
            dtb_processing_params = yaml.safe_load(file)

        database_params = dtb_processing_params["database_params"]
        dtb_type = database_params["database_type"]
        
        data_processing = dtb_processing_params["data_processing"]
        self.remove_N2 = not data_processing["with_N_chemistry"]
        self.log_transform_Y = data_processing["log_transform_Y"]

        data_clustering = dtb_processing_params["data_clustering"]
        self.clustering_type = data_clustering["clustering_method"]
        self.nb_clusters = data_clustering["nb_clusters"]

        if dtb_type == "stoch":
            params_file = "dtb_params.yaml"
            prefix = "STOCH"
        elif dtb_type == "flamelets":
            params_file = "dtb_params_flmts.yaml"
            prefix = "FLAMELETS"

        with open(os.path.join(self.run_folder, f"{prefix}_DTB_{database_params['dtb_folder_suffix']}",params_file), "r") as file:
            dtb_parameters = yaml.safe_load(file)

        self.fuel = dtb_parameters["fuel"]
        self.mechanism = dtb_parameters["mech_file"]

        # Model folder name
        self.model_name = "MODEL_" + networks_parameters["model_name_suffix"]
        self.new_model_folder = networks_parameters["new_model_folder"]
        # Network(s) structure (1 per cluster)
        self.networks_types = networks_parameters["networks_types"] # list[str] (string in model_type.keys()), 1 per cluster
        self.networks_defs = networks_parameters["networks_def"] # list[str], keys to network parameters in "clusters", 1 per cluster
        self.clusters = networks_parameters["clusters"]
        self.learning = networks_parameters["learning"] # DA to CM: same learning for all clusters ?
        self.learning["optimizer"] = networks_parameters.get("optimizer", "adam")
        self.learning["loss"] = networks_parameters.get("loss", "MSE")
        self.learning["scheduler_option"] = networks_parameters.get("scheduler_option", "ExpLR")

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

        # Get the number of clusters   CM: TO ADAPT !!!
        # assert (self.nb_clusters == len(next(os.walk(self.dataset_path))[1]))
        # if ((self.nb_clusters != len(self.networks_defs)) or (self.nb_clusters != len(self.networks_types))):
        #     raise ValueError(
        #         f"number of clusters in {self.dataset_path} ({self.nb_clusters}) inconsistent with "
        #         f"'networks_types' and/or 'networks_files' length ({len(self.networks_defs)} and {len(self.networks_types)})"
        #     )
        print("CLUSTERING:")
        print(f">> Number of clusters is: {self.nb_clusters}")

        # We copy the mechanism files in order to use them for testing
        if self.new_model_folder:
            shutil.copy(self.mechanism, self.directory)
        
        shutil.copy(os.path.join(self.run_folder, "networks_params.yaml"), self.directory)

        gas = ct.Solution(self.mechanism)
        self.A_element = utils.get_molar_mass_atomic_matrix(gas.species_names, self.fuel, not self.remove_N2)

        # Training stats
        if self.new_model_folder:
            os.mkdir(self.directory + "/training")
            os.mkdir(self.directory + "/evaluation" )

            for i_cluster in range(self.nb_clusters):
                os.mkdir(f"{self.directory}/training/cluster_{i_cluster}")
                os.mkdir(f"{self.directory}/training/cluster_{i_cluster}/training_curves/")
                os.mkdir(f"{self.directory}/evaluation/cluster_{i_cluster}")

            if (self.nb_clusters > 1):
                # Adding copies of clustering parameters for later use in inference
                self.copy_clusterer()

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
                layers_activation_list = [activation_functions[act] for act in network_parameters["layers_activation_list"]]
                layers_type = network_parameters["layers_type"]
            
                model = model_type[network_type](self.device, nb_units_in_layers_list, layers_type, layers_activation_list)
            elif ("DeepONet" in network_type):
                # Network shapes
                nb_units_in_layers_list = copy.deepcopy(network_parameters["nb_units_in_layers_list"])
                layers_activation_list = {}
                for k in network_parameters["layers_activation_list"].keys():
                    layers_activation_list[k] = [activation_functions[act] for act in network_parameters["layers_activation_list"][k]]
                layers_type = network_parameters["layers_type"]
                n_neurons = network_parameters["n_neurons"]
                nb_units_in_layers_list["branch"].insert(0,n_in)
                nb_units_in_layers_list["branch"].append(n_neurons*n_out)
                nb_units_in_layers_list["trunk"].insert(0,1)
                nb_units_in_layers_list["trunk"].append(n_neurons*n_out)
                model = model_type[network_type](self.device, nb_units_in_layers_list, layers_type, layers_activation_list, n_out, n_neurons)
            elif ("DeepONetShift" in network_type):
                # Network shapes
                nb_units_in_layers_list = copy.deepcopy(network_parameters["nb_units_in_layers_list"])
                layers_activation_list = {}
                for k in network_parameters["layers_activation_list"].keys():
                    layers_activation_list[k] = [activation_functions[act] for act in network_parameters["layers_activation_list"][k]]
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
    

    def _inverse_scale(self, scaled_tensor, mean, std, log_transform: bool):
        """Inverse scaling helper for mass fractions.

        scaled_tensor: tensor in scaled space
        mean, std: scaling statistics (tensors on same device)
        log_transform: whether data were log-transformed
        """
        y = mean + (std + 1e-7) * scaled_tensor
        if log_transform:
            y = torch.exp(y)
        return y

    def train_model(self, i_cluster, model, loss_fn, optimizer, scheduler, X_train, X_val, Y_train, Y_val, Yscaler_mean, Yscaler_std):

        X_train = torch.tensor(X_train, dtype=torch.float64)
        Y_train = torch.tensor(Y_train, dtype=torch.float64)
        X_val = torch.tensor(X_val, dtype=torch.float64)
        Y_val = torch.tensor(Y_val, dtype=torch.float64)
        A_element = torch.tensor(self.A_element, dtype=torch.float64)
        Yscaler_mean = torch.from_numpy(Yscaler_mean).to(torch.float64)
        Yscaler_std = torch.from_numpy(Yscaler_std).to(torch.float64)

        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)
        X_val = X_val.to(self.device)
        Y_val = Y_val.to(self.device)

        A_element = A_element.to(self.device)    # Array to store the loss and validation loss

        Yscaler_mean = Yscaler_mean.to(self.device)
        Yscaler_std = Yscaler_std.to(self.device)

        n_epochs = self.learning["epochs_list"][i_cluster]

        loss_list = np.empty(n_epochs)

        # Validation / conservation computed every "val_every" epochs (default 10)
        val_every = int(self.learning.get("val_every", 10))
        n_val_points = max(1, n_epochs // val_every)

        # Array to store sum of mass fractions: mean, min and max
        stats_sum_yk = np.empty((n_val_points, 3))

        # Array to store elements conservation: mean, min and max
        stats_A_elements = np.empty((n_val_points, A_element.shape[0], 3))

        epochs = np.arange(n_epochs)
        epochs_small = epochs[::val_every]

        best_val_loss = float("inf")
        best_state_dict = None

        val_loss_list = np.empty(n_val_points)

        for epoch in range(n_epochs):

            # Training parameters
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, len(X_train), self.learning["batch_size"]):

                Xbatch = X_train[i:i+self.learning["batch_size"]]
                y_pred = model(Xbatch)
                ybatch = Y_train[i:i+self.learning["batch_size"]]
                loss = loss_fn(y_pred, ybatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
                # print("Targets stats:", Y_train.min().item(), Y_train.max().item())
                # print("Preds stats:", y_pred.min().item(), y_pred.max().item())
                # input("Press Enter to continue.")

            loss_list[epoch] = epoch_loss / max(1, n_batches)

            before_lr = optimizer.param_groups[0]["lr"]
            if self.learning["scheduler_option"]!="None":
                scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]

            # Computing validation loss and mass conservation metric (only every "val_every" epochs as it is expensive)
            if epoch % val_every == 0:
                model.eval()  # evaluation mode
                with torch.no_grad():

                    # VALIDATION LOSS
                    y_val_pred = model(X_val)
                    val_loss = loss_fn(y_val_pred, Y_val)

                    # SUM OF MASS FRACTION
                    # Inverse scale done by hand to stay with Torch arrays
                    yk = self._inverse_scale(y_val_pred, Yscaler_mean, Yscaler_std, self.log_transform_Y)
                    sum_yk = yk.sum(axis=1)
                    sum_yk = sum_yk.detach().cpu().numpy()
                    idx = epoch // val_every
                    stats_sum_yk[idx,0] = sum_yk.mean() 
                    stats_sum_yk[idx,1] = sum_yk.min()
                    stats_sum_yk[idx,2] = sum_yk.max()

                    # ELEMENTS CONSERVATION
                    yval_in = self._inverse_scale(X_val[:,1:-1], Yscaler_mean, Yscaler_std, self.log_transform_Y)
                    ye_in = torch.matmul(A_element, torch.transpose(yval_in, 0, 1))
                    ye_out = torch.matmul(A_element, torch.transpose(yk, 0, 1))
                    delta_ye = (ye_out - ye_in)/(ye_in+1e-10)
                    delta_ye = delta_ye.detach().cpu().numpy()
                    stats_A_elements[idx, :, 0] = delta_ye.mean(axis=1)
                    stats_A_elements[idx, :, 1] = delta_ye.min(axis=1)
                    stats_A_elements[idx, :, 2] = delta_ye.max(axis=1)

                model.train()   # Back to training mode

                if best_state_dict is None or val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_state_dict = copy.deepcopy(model.state_dict())

                val_loss_list[idx] = val_loss.item()

            print(f"Finished epoch {epoch}")
            print(f"    >> lr: {before_lr} -> {after_lr}")
            print(f"    >> Loss (mean over batches): {loss_list[epoch]}")
            if epoch % val_every==0:
                print(f"    >> Validation loss: {val_loss}")

        # Restore best model (based on validation loss) before returning
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

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
            opt_name = str(self.learning.get("optimizer", "adam")).lower()
            if opt_name == "adam":
                optimizer = optim.Adam(model.parameters(), lr=self.learning["initial_learning_rate"])
            elif opt_name == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=self.learning["initial_learning_rate"])
            else:
                raise ValueError(f"Unsupported optimizer '{opt_name}'")

            loss_name = str(self.learning.get("loss", "mse")).lower()
            if loss_name == "mse":
                loss_fn = nn.MSELoss()
            elif loss_name == "l1":
                loss_fn = nn.L1Loss()
            else:
                raise ValueError(f"Unsupported loss '{loss_name}'")

            # Set scheduler
            scheduler_option = self.learning["scheduler_option"]
            if scheduler_option=="ExpLR":
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.learning["decay_rate"])
            elif scheduler_option == "None" or scheduler_option is None:
                # Identity scheduler (no LR change)
                scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
            else:
                raise ValueError("Only scheduler implemented yet is ExpLR or None")
            
            # Perform training
            epochs, epochs_small, loss_list, val_loss_list, stats_sum_yk, stats_A_elements = self.train_model(i_cluster, model, loss_fn, optimizer, scheduler, X_train, X_val, Y_train, Y_val, Yscaler_mean, np.sqrt(Yscaler_var))

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

        plt.savefig( f"{self.directory}/training/cluster_{i_cluster}/training_curves/loss.png")

        # MASS CONSERVATION
        fig, ax = plt.subplots()

        ax.plot(epochs_small, stats_sum_yk[:,0], color="k")
        ax.plot(epochs_small, stats_sum_yk[:,1], color="k", ls="--")
        ax.plot(epochs_small, stats_sum_yk[:,2], color="k", ls="--")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\sum_k \ Y_k$")

        plt.savefig( f"{self.directory}/training/cluster_{i_cluster}/training_curves/mass_conservation.png")

        # ELEMENTS CONSERVATION
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

        # C
        ax1.plot(epochs_small, 100*stats_A_elements[:,0,0], color="k")
        ax1.plot(epochs_small, 100*stats_A_elements[:,0,1], color="k", ls="--")
        ax1.plot(epochs_small, 100*stats_A_elements[:,0,2], color="k", ls="--")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel(r"$\Delta Y_C$ $(\%$)")

        # H
        ax2.plot(epochs_small, 100*stats_A_elements[:,1,0], color="k")
        ax2.plot(epochs_small, 100*stats_A_elements[:,1,1], color="k", ls="--")
        ax2.plot(epochs_small, 100*stats_A_elements[:,1,2], color="k", ls="--")

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel(r"$\Delta Y_H$ $(\%)$")

        # O
        ax3.plot(epochs_small, 100*stats_A_elements[:,2,0], color="k")
        ax3.plot(epochs_small, 100*stats_A_elements[:,2,1], color="k", ls="--")
        ax3.plot(epochs_small, 100*stats_A_elements[:,2,2], color="k", ls="--")

        ax3.set_xlabel("Epoch")
        ax3.set_ylabel(r"$\Delta Y_O$ $(\%)$")

        # N
        ax4.plot(epochs_small, 100*stats_A_elements[:,3,0], color="k")
        ax4.plot(epochs_small, 100*stats_A_elements[:,3,1], color="k", ls="--")
        ax4.plot(epochs_small, 100*stats_A_elements[:,3,2], color="k", ls="--")

        ax4.set_xlabel("Epoch")
        ax4.set_ylabel(r"$\Delta Y_N$ $(\%)$")

        fig.tight_layout()

        plt.savefig( f"{self.directory}/training/cluster_{i_cluster}/training_curves/elements_conservation.png")

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
        expected dense layer formats.
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
            # Save each top-level layer directly, no 'model' group
            for layer_name, module in model.model.named_children():  # note: model.model is nn.Sequential
                save_module(f, module, layer_name, getattr(model, 'activation_map', {}))
