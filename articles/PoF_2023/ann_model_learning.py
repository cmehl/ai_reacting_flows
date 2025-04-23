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
networks_parameters["networks_def"] = ["cluster0", "cluster1"]

networks_parameters["clusters"] = {}
networks_parameters["clusters"]["cluster0"] = {}
networks_parameters["clusters"]["cluster0"]["nb_units_in_layers_list"] = [20,20]   # Network shape: number of units in each hidden layer
networks_parameters["clusters"]["cluster0"]["layers_activation_list"] = ['tanh','tanh','Id']    # Activation functions
networks_parameters["clusters"]["cluster0"]["layers_type"] = ["dense","dense","dense"] # "dense" or "resnet"
networks_parameters["clusters"]["cluster1"] = {}
networks_parameters["clusters"]["cluster1"]["nb_units_in_layers_list"] = [40,40]   # Network shape: number of units in each hidden layer
networks_parameters["clusters"]["cluster1"]["layers_activation_list"] = ['tanh','tanh','Id']    # Activation functions
networks_parameters["clusters"]["cluster1"]["layers_type"] = ["dense","dense","dense"] # "dense" or "resnet"

networks_parameters["learning"] = {}
networks_parameters["learning"]["initial_learning_rate"] = 1.0e-3           # Initial learnign rate
networks_parameters["learning"]["batch_size"] = 2048         # Batch size for the gradient descent
networks_parameters["learning"]["epochs_list"] = [500,500]    # number of epochs for each cluster
networks_parameters["learning"]["decay_rate"] = 0.9991        # Exponential decay: rate

with open("networks_params.yaml", "w") as file:
    yaml.dump(networks_parameters, file, default_flow_style=False)

mlp_model = NN_manager()
mlp_model.train_all_clusters()