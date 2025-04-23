import sys
import os
import shutil
import joblib

import numpy as np
import pandas as pd
import oyaml as yaml
import cantera as ct
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import ai_reacting_flows.tools.utilities as utils
from ai_reacting_flows.ann_model_generation.NN_models import MLPModel, DeepONet, DeepONet_shift

torch.set_default_dtype(torch.float64)

activation_functions = {"ReLU": nn.ReLU, "GeLU" : nn.GELU, "tanh" : nn.Tanh, "Id" : nn.Identity}
model_type = {"MLP": MLPModel, "DeepONet": DeepONet, "DeepONetShift": DeepONet_shift}

class NN_manager():
    def __init__(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
        self.run_folder = os.getcwd()
        with open(os.path.join(self.run_folder, "dtb_params.yaml"), "r") as file: # DA to CM: should be the same file for previous steps
            dtb_parameters = yaml.safe_load(file)
        #
        # Database and data processing related inputs
        #
        self.fuel = dtb_parameters["fuel"]
        self.mechanism = dtb_parameters["mechanism"]
        # Data processing
        self.dataset_path = dtb_parameters["dataset_path"] # DA to CM: name should be common with previous steps
        self.remove_N2 = dtb_parameters["remove_N2"]
        self.log_transform_Y = dtb_parameters["log_transform_Y"]
        self.clustering_type = dtb_parameters["clusterization_method"]

        #
        # Networks structures parameters
        #
        with open(os.path.join(self.run_folder, "networks_params.yaml"), "r") as file:
            networks_parameters = yaml.safe_load(file)
        # Model folder name
        self.model_name = "MODEL_" + networks_parameters["model_name_suffix"]
        self.new_model_folder = networks_parameters["new_model_folder"]
        # Network(s) structure (1 per cluster)
        self.networks_types = networks_parameters["networks_types"] # list[str] (string in model_type.keys()), 1 per cluster
        self.networks_defs = networks_parameters["networks_def"] # list[str], keys to network parameters in "clusters", 1 per cluster
        self.clusters = networks_parameters["clusters"]
        self.learning = networks_parameters["learning"] # DA to CM: same learning for all clusters ?
        self.learning["optimizer"] = "adam"
        self.learning["loss"] = "MSE"
        self.learning["scheduler_option"] = "ExpLR"

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
                sys.exit(f"ERROR: new_model_folder is set to False but model {self.directory} does not exist")
            print(f">> Existing model folder {self.directory} is used. \n")

        # Get the number of clusters
        self.nb_clusters = len(next(os.walk(self.dataset_path))[1])
        if ((self.nb_clusters != len(self.networks_defs)) or (self.nb_clusters != len(self.networks_types))):
            sys.exit((f"ERROR: number of clusters in {self.dataset_path} ({self.nb_clusters}) inconsistent with"
                     f" \"networks_types\" and/or \"networks_files \" length ({len(self.networks_defs)} and {len(self.networks_types)})"))
        print("CLUSTERING:")
        print(f">> Number of clusters is: {self.nb_clusters}")

        # We copy the mechanism files in order to use them for testing
        if self.new_model_folder:
            shutil.copy(self.mechanism, self.directory)

        gas = ct.Solution(self.mechanism)
        self.A_element = utils.get_molar_mass_atomic_matrix(gas.species_names, self.fuel, not self.remove_N2)

        # Training stats
        if self.new_model_folder:
            os.mkdir(self.directory + "/training")
            os.mkdir(self.directory + "/evaluation" )

            if (self.nb_clusters > 1):
                for i_cluster in range(self.nb_clusters):
                    os.mkdir(f"{self.directory}/training/cluster_{i_cluster}")
                    os.mkdir(f"{self.directory}/training/cluster_{i_cluster}/training_curves/")
                    os.mkdir(f"{self.directory}/evaluation/cluster_{i_cluster}")
            else:
                os.mkdir(self.directory + "/training/training_curves")

            # Adding copies of clustering parameters for later use in inference
            self.copy_clusterer()

        # Saving parameters for later use in testing
        self.save_dtb_params()
        self.save_networks_params()

    def create_model(self, i_cluster, X_val, Y_val):
        network_type = self.networks_types[i_cluster]
        if (network_type not in model_type.keys()):
            sys.exit(f"ERROR: network_type \"{network_type}\" does not exist")
        else:
            network_parameters = self.clusters[self.networks_defs[i_cluster]]
            if (network_type == "MLP"):
                # Network shapes
                nb_units_in_layers_list = network_parameters["nb_units_in_layers_list"]
                nb_units_in_layers_list.insert(0, X_val.shape[1])
                nb_units_in_layers_list.append(Y_val.shape[1])
                layers_activation_list = [activation_functions[act] for act in network_parameters["layers_activation_list"]]
                layers_type = network_parameters["layers_type"]
            
                model = model_type[network_type](self.device, nb_units_in_layers_list, layers_type, layers_activation_list)
            elif ("DeepONet" in network_type):
                n_in = X_val.shape[1]
                n_out = Y_val.shape[1]
                # Network shapes
                nb_units_in_layers_list = network_parameters["nb_units_in_layers_list"]
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
                n_in = X_val.shape[1]
                n_out = Y_val.shape[1]
                # Network shapes
                nb_units_in_layers_list = network_parameters["nb_units_in_layers_list"]
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

    def train_model(self, i_cluster, model, loss_fn, optimizer, scheduler, X_train, X_val, Y_train, Y_val, Yscaler_mean, Yscaler_std):
        X_train = torch.tensor(X_train.values, dtype=torch.float64)
        Y_train = torch.tensor(Y_train.values, dtype=torch.float64)
        X_val = torch.tensor(X_val.values, dtype=torch.float64)
        Y_val = torch.tensor(Y_val.values, dtype=torch.float64)
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
        val_loss_list = np.empty(n_epochs//10)

        # Array to store sum of mass fractions: mean, min and max
        stats_sum_yk = np.empty((n_epochs//10,3))

        # Array to store elements conservation: mean, min and max
        stats_A_elements = np.empty((n_epochs//10,A_element.shape[0],3))

        epochs = np.arange(n_epochs)
        epochs_small = epochs[::10]

        for epoch in range(n_epochs):

            # Training parameters
            for i in range(0, len(X_train), self.learning["batch_size"]):

                Xbatch = X_train[i:i+self.learning["batch_size"]]
                y_pred = model(Xbatch)
                ybatch = Y_train[i:i+self.learning["batch_size"]]
                loss = loss_fn(y_pred, ybatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("Targets stats:", Y_train.min().item(), Y_train.max().item())
                # print("Preds stats:", y_pred.min().item(), y_pred.max().item())
                # input("Press Enter to continue.")

            loss_list[epoch] = loss.item()

            before_lr = optimizer.param_groups[0]["lr"]
            if self.learning["scheduler_option"]!="None":
                scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]

            # Computing validation loss and mass conservation metric (only every 10 epochs as it is expensive)
            if epoch%10==0:
                model.eval()  # evaluation mode
                with torch.no_grad():

                    # VALIDATION LOSS
                    y_val_pred = model(X_val)
                    val_loss = loss_fn(y_val_pred, Y_val)

                    # SUM OF MASS FRACTION
                    #Inverse scale done by hand to stay with Torch arrays
                    yk = Yscaler_mean + (Yscaler_std + 1e-7)*y_val_pred
                    if self.log_transform_Y:
                        yk = torch.exp(yk)
                    sum_yk = yk.sum(axis=1)
                    sum_yk = sum_yk.detach().cpu().numpy()
                    stats_sum_yk[epoch//10,0] = sum_yk.mean() 
                    stats_sum_yk[epoch//10,1] = sum_yk.min()
                    stats_sum_yk[epoch//10,2] = sum_yk.max()

                    # ELEMENTS CONSERVATION
                    yval_in = Yscaler_mean + (Yscaler_std + 1e-7)*X_val[:,1:]
                    if self.log_transform_Y:
                        yval_in = torch.exp(yval_in)
                    ye_in = torch.matmul(A_element, torch.transpose(yval_in, 0, 1))
                    ye_out = torch.matmul(A_element, torch.transpose(yk, 0, 1))
                    delta_ye = (ye_out - ye_in)/(ye_in+1e-10)
                    delta_ye = delta_ye.detach().cpu().numpy()
                    stats_A_elements[epoch//10, :, 0] = delta_ye.mean(axis=1)
                    stats_A_elements[epoch//10, :, 1] = delta_ye.min(axis=1)
                    stats_A_elements[epoch//10, :, 2] = delta_ye.max(axis=1)

                model.train()   # Back to training mode
                val_loss_list[epoch//10] = val_loss

            print(f"Finished epoch {epoch}")
            print(f"    >> lr: {before_lr} -> {after_lr}")
            print(f"    >> Loss: {loss}")
            if epoch%10==0:
                print(f"    >> Validation loss: {val_loss}")

        return epochs, epochs_small, loss_list, val_loss_list, stats_sum_yk, stats_A_elements

    def train_all_clusters(self):
        for i_cluster in range(self.nb_clusters):
            X_train, X_val, Y_train, Y_val = self.read_training_data(i_cluster)            
            Xscaler_mean, Xscaler_var, Yscaler_mean, Yscaler_var = self.get_scalers_stats(i_cluster)
            model = self.create_model(i_cluster, X_val, Y_val)
            optimizer = optim.Adam(model.parameters(), lr=self.learning["initial_learning_rate"])
            loss_fn = nn.MSELoss()
            # if self.scheduler_option=="ExpLR": # DA: not needed yet (only 1 implemented)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.learning["decay_rate"])
            
            epochs, epochs_small, loss_list, val_loss_list, stats_sum_yk, stats_A_elements = self.train_model(i_cluster, model, loss_fn, optimizer, scheduler, X_train, X_val, Y_train, Y_val, Yscaler_mean, np.sqrt(Yscaler_var))

            self.plot_losses_conservation(i_cluster, epochs, epochs_small, loss_list, val_loss_list, stats_sum_yk, stats_A_elements)

            torch.save(model, os.path.join(self.directory, f"cluster{i_cluster}_model.pth"))
            
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
        fig, ((ax1, ax2)) = plt.subplots(1,2)

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
                sys.exit("ERROR: nb_cluster > 1 but cluster_type undefined (should be 'progvar' or 'kmeans')")

    def read_training_data(self, i_cluster):

        X_train = pd.read_csv(filepath_or_buffer= f"{self.dataset_path:s}/cluster{i_cluster}/X_train.csv")
        Y_train = pd.read_csv(filepath_or_buffer= f"{self.dataset_path:s}/cluster{i_cluster}/Y_train.csv")
            
        X_val = pd.read_csv(filepath_or_buffer= f"{self.dataset_path:s}/cluster{i_cluster}/X_val.csv")
        Y_val = pd.read_csv(filepath_or_buffer= f"{self.dataset_path:s}/cluster{i_cluster}/Y_val.csv")

        return X_train, X_val, Y_train, Y_val
    
    def read_scalers(self, i_cluster):
        Xscaler = joblib.load(os.path.join(f"{self.dataset_path:s}/cluster{i_cluster}", "Xscaler.save"))
        Yscaler = joblib.load(os.path.join(f"{self.dataset_path:s}/cluster{i_cluster}", "Yscaler.save"))

        return Xscaler, Yscaler
    
    def get_scalers_stats(self, i_cluster):
        Xscaler = joblib.load(os.path.join(f"{self.dataset_path:s}/cluster{i_cluster}", "Xscaler.save"))
        Yscaler = joblib.load(os.path.join(f"{self.dataset_path:s}/cluster{i_cluster}", "Yscaler.save"))

        return Xscaler.mean_, Xscaler.var_, Yscaler.mean_, Yscaler.var_

    def save_scalers(self, i_cluster, Xscaler, Yscaler):

        joblib.dump(Xscaler, os.path.join(f"{self.directory:s}/cluster{i_cluster}", "Xscaler.save"))
        joblib.dump(Yscaler, os.path.join(f"{self.directory:s}/cluster{i_cluster}", "Yscaler.save"))
    
    def save_dtb_params(self):
        data = {
            "log_transform_Y" : self.log_transform_Y,
            "remove_N2" : self.remove_N2,
            "mechanism" : self.mechanism,
            "dataset_path" : self.dataset_path,
            "fuel" : self.fuel,
            "clusterization_method" : self.clustering_type
        }

        with open(os.path.join(self.directory,"dtb_params.yaml"), "w") as file:
            yaml.dump(data, file, default_flow_style=False)

    def save_networks_params(self):
        data = {
            "model_name_suffix" : self.model_name[6:],
            "new_model_folder" : self.new_model_folder,
            "networks_types" : self.networks_types,
            "networks_defs" : self.networks_defs,
            "clusters" : self.clusters,
            "learning" : self.learning
        }

        with open(os.path.join(self.directory,"networks_params.yaml"), "w") as file:
            yaml.dump(data, file, default_flow_style=False)

    # def read_learning_config(self):
        # relics from previous TensorFlow version of ARF --> might be re-integrated later
        # self.use_final_lr = learning_parameters["use_final_lr"]
        # if (self.use_final_lr):
        #     self.final_learning_rate = learning_parameters["final_learning_rate"]
        # Parameters of the exponential decay schedule (learning rate decay)
        #
        # self.decay_steps = learning_parameters["decay_steps"]
        # self.decay_rate = learning_parameters["decay_rate"]
        # self.staircase = learning_parameters["staircase"]