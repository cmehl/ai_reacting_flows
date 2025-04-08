import oyaml as yaml
import os

from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager

# Dictionary with parameters
database_parameters = {}
database_parameters["fuel"] = "H2"           # Fuel
database_parameters["mechanism"] = "../../data/chemical_mechanisms/mech_H2.yaml"

run_folder = os.getcwd()
database_parameters["dataset_path"] = os.path.join(run_folder,"STOCH_DTB_H2_POF_2023","dtb_resampled")    # path of the database
database_parameters["remove_N2"] = True    # if True, N2 is removed from the neural network prediction
database_parameters["log_transform_Y"] = 1
database_parameters["clusterization_method"] = "kmeans"  # "progvar" or "kmeans"

with open("dtb_params.yaml", "w") as file:
    yaml.dump(database_parameters, file, default_flow_style=False)
    
networks_parameters = {}

networks_parameters["model_name_suffix"] = "H2_POF_2023" # Name of the resulting model folder (as a suffix of MODEL)
networks_parameters["new_model_folder"] = True # New model folders are training models inside existing folder
networks_parameters["networks_types"] = ["MLP", "MLP"]
networks_parameters["networks_files"] = ["network_cluster0.yaml", "network_cluster1.yaml"]

with open("networks_params.yaml", "w") as file:
    yaml.dump(networks_parameters, file, default_flow_style=False)

network_cluster0 = {}

network_cluster0["nb_units_in_layers_list"] = [20,20]   # Network shape: number of units in each hidden layer
network_cluster0["layers_activation_list"] = ['tanh','tanh','Id']    # Activation functions
network_cluster0["layers_type"] = ["dense","dense","dense"] # "dense" or "resnet"

with open("network_cluster0.yaml", "w") as file:
    yaml.dump(network_cluster0, file, default_flow_style=False)

network_cluster1 = {}

network_cluster1["nb_units_in_layers_list"] = [40,40]   # Network shape: number of units in each hidden layer
network_cluster1["layers_activation_list"] = ['tanh','tanh','Id']    # Activation functions
network_cluster1["layers_type"] = ["dense","dense","dense"] # "dense" or "resnet"

with open("network_cluster1.yaml", "w") as file:
    yaml.dump(network_cluster1, file, default_flow_style=False)

learning_parameters = {}

learning_parameters["initial_learning_rate"] = 1.0e-3           # Initial learnign rate
learning_parameters["batch_size"] = 2048         # Batch size for the gradient descent
learning_parameters["epochs_list"] = [500,500]    # number of epochs for each cluster
learning_parameters["decay_rate"] = 0.9991        # Exponential decay: rate

with open("learning_config.yaml", "w") as file:
    yaml.dump(learning_parameters, file, default_flow_style=False)

mlp_model = NN_manager()
mlp_model.train_all_clusters()