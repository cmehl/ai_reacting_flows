import sys
import os
import shutil
import joblib

import numpy as np
import pandas as pd
import oyaml as yaml
import cantera as ct

import torch
import torch.nn as nn
import torch.optim as optim

import ai_reacting_flows.tools.utilities as utils
from NN_models import MLPModel, DeepONet, DeepONet_shift

torch.set_default_dtype(torch.float64)

activation_functions = {"ReLU": nn.ReLU, "GeLU" : nn.GELU, "tanh" : nn.Tanh}
model_type = {"MLP": MLPModel, "DeepONet": DeepONet, "DeepONetShift": DeepONet_shift}

class NN_manager(nn.Module):
    def __init__(self):
        super.__init__()
        
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
        self.networks_files = networks_parameters["networks_files"] # list[str], path to network parameters, 1 per cluster

        #
        # Learning/Optimization parameters
        #
        self.read_learning_config(self.run_folder) # DA to CM: same for all clusters ?

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
        if ((self.nb_clusters != len(self.networks_files)) or (self.nb_clusters != len(self.networks_files))):
            sys.exit((f"ERROR: number of clusters in {self.dataset_path} ({self.nb_clusters}) inconsistent with"
                     f" \"networks_types\" and/or \"networks_files \" length ({len(self.networks_files)} and {len(self.networks_files)})"))
        print("CLUSTERING:")
        print(f">> Number of clusters is: {self.nb_clusters}")

        # Defining mechanism file (either detailed or reduced)
        self.mechanism = os.path.join(self.dataset_path, self.mechanism)

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
        self.save_learning_config()

    def create_model(self, i_cluster):
        network_type = self.networks_types[i_cluster]
        if (network_type not in model_type.keys()):
            sys.exit(f"ERROR: network_type \"{network_type}\" does not exist")
        else:
            network_file = self.networks_files[i_cluster]
            with open(os.path.join(self.run_folder, network_file), "r") as file:
                network_parameters = yaml.safe_load(file)
            if (network_type == "MLP"):
                # Network shapes
                nb_units_in_layers_list = network_parameters["nb_units_in_layers_list"]
                layers_activation_list = [activation_functions[act] for act in network_parameters["layers_activation_list"]]
                layers_type = network_parameters["layers_type"]
            
                model = model_type[network_type](nb_units_in_layers_list, layers_type, layers_activation_list)
            elif ("DeepONet" in network_type):
                # Network shapes
                nb_units_in_layers_list = network_parameters["nb_units_in_layers_list"]
                layers_activation_list = {}
                for k in network_parameters["layers_activation_list"].keys():
                    layers_activation_list[k] = [activation_functions[act] for act in network_parameters["layers_activation_list"][k]]
                layers_type = network_parameters["layers_type"]
                n_neurons = network_parameters["n_neurons"]
                model = model_type[network_type](nb_units_in_layers_list, layers_type, layers_activation_list, n_neurons)
        
        return model

    def train_model(self, i_cluster, model, loss_fn, optimizer, scheduler):
        X_train, X_val, Y_train, Y_val = self.read_training_data(i_cluster)
        Yscaler_mean, Yscaler_std = self.get_Yscaler_stats(i_cluster)

        X_train = torch.tensor(X_train, dtype=torch.float64)
        Y_train = torch.tensor(Y_train, dtype=torch.float64)
        X_val = torch.tensor(X_val, dtype=torch.float64)
        Y_val = torch.tensor(Y_val, dtype=torch.float64)
        A_element = torch.tensor(self.A_element, dtype=torch.float64)
        Yscaler_mean = torch.from_numpy(Yscaler_mean)
        Yscaler_std = torch.from_numpy(Yscaler_std)

        X_train = X_train.to(self.device)
        Y_train = Y_train.to(self.device)
        X_val = X_val.to(self.device)
        Y_val = Y_val.to(self.device)

        A_element = A_element.to(self.device)    # Array to store the loss and validation loss

        Yscaler_mean = Yscaler_mean.to(self.device)
        Yscaler_std = Yscaler_std.to(self.device)

        n_epochs = self.epochs_list[i_cluster]

        loss_list = np.empty(n_epochs)
        val_loss_list = np.empty(n_epochs//10)

        # Array to store sum of mass fractions: mean, min and max
        stats_sum_yk = np.empty((n_epochs//10,3))

        # Array to store elements conservation: mean, min and max
        stats_A_elements = np.empty((n_epochs//10,4,3))

        epochs = np.arange(n_epochs)
        epochs_small = epochs[::10]

        for epoch in range(n_epochs):

            # Training parameters
            for i in range(0, len(X_train), self.batch_size):

                Xbatch = X_train[i:i+self.batch_size]
                y_pred = model(Xbatch)
                ybatch = Y_train[i:i+self.batch_size]
                loss = loss_fn(y_pred, ybatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_list[epoch] = loss

            before_lr = optimizer.param_groups[0]["lr"]
            if self.scheduler_option!="None":
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
                    yval_in = Yscaler_mean + (Yscaler_std + 1e-7)*X_val[:,1:-1]
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
            model = self.create_model(i_cluster)
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr_ini)
            loss_fn = nn.MSELoss()
            # if self.scheduler_option=="ExpLR": # DA: not needed yet (only 1 implemented)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma_lr)
            
            self.train_model(i_cluster, model, loss_fn, optimizer, scheduler)

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

        Xscaler = joblib.load(os.path.join(f"{self.dataset_path:s}/cluster{i_cluster}", "Xscaler.pkl"))
        Yscaler = joblib.load(os.path.join(f"{self.dataset_path:s}/cluster{i_cluster}", "Yscaler.pkl"))

        return Xscaler, Yscaler
    
    def get_Yscaler_stats(self, i_cluster):
        Yscaler = joblib.load(os.path.join(f"{self.dataset_path:s}/cluster{i_cluster}", "processed_database", "Yscaler.pkl"))

        return Yscaler.mean, Yscaler.std

    def save_scalers(self, i_cluster, Xscaler, Yscaler):

        joblib.dump(Xscaler, os.path.join(f"{self.directory:s}/cluster{i_cluster}", "Xscaler.pkl"))
        joblib.dump(Yscaler, os.path.join(f"{self.directory:s}/cluster{i_cluster}", "Yscaler.pkl"))
    
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
            "networks_files" : self.networks_files
        }

        with open(os.path.join(self.directory,"dtb_params.yaml"), "w") as file:
            yaml.dump(data, file, default_flow_style=False)

    def read_learning_config(self, filepath):
        with open(os.path.join(filepath, "learning_config.yaml"), "r") as file:
            learning_parameters = yaml.safe_load(file)

        self.lr_ini = learning_parameters["initial_learning_rate"]
        self.batch_size = learning_parameters["batch_size"]
        self.scheduler_option = "ExpLR"
        self.gamma_lr = learning_parameters["gamma_lr"]
        self.epochs_list = learning_parameters["epochs_list"]

        # relics from previous TensorFlow version of ARF --> might be re-integrated later
        # self.use_final_lr = learning_parameters["use_final_lr"]
        # if (self.use_final_lr):
        #     self.final_learning_rate = learning_parameters["final_learning_rate"]
        # Parameters of the exponential decay schedule (learning rate decay)
        #
        # self.decay_steps = learning_parameters["decay_steps"]
        # self.decay_rate = learning_parameters["decay_rate"]
        # self.staircase = learning_parameters["staircase"]

    def save_learning_config(self):

        data = {
            "initial_learning_rate": self.lr_ini,
            "batch_size": self.batch_size,
            "epochs_list": self.epochs_list,
            "scheduler_option": "ExpLR",
            "gamma_lr": self.gamma_lr,
            "optimizer": "adam",
            "loss": "MSE"
        }

        with open(os.path.join(self.directory,"learning_config.yaml"), "w") as file:
            yaml.dump(data, file, default_flow_style=False)