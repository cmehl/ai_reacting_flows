# This is a script version of the notebook ann_model_learning.ipynb in the notebooks directory

from ai_reacting_flows.ann_model_generation.MLP_model import MLPModel


# Dictionary with parameters
training_parameters = {}

training_parameters["model_name_suffix"] = "TEST"    # Name of the resulting model folder (as a suffix of MODEL)
    
training_parameters["dataset_path"] = "/work/mehlc/2_IA_KINETICS/ai_reacting_flows/scripts/STOCH_DTB_hotspot_H2_DEV/database_1"    # path of the database
    
training_parameters["dt_simu"] = 5.0e-07     # Time step of the prediction
    
training_parameters["fuel"] = "H2"           # Fuel
training_parameters["mechanism_type"] = "reduced"    # Mechanism type for ANN chemistry: detailed or reduced

training_parameters["remove_N2"] = True    # if True, N2 is removed from the neural network prediction

training_parameters["nb_units_in_layers_list"] = [[20,20], [10,10], [10,10]]   # Network shape: number of units in each layer
training_parameters["layers_activation_list"] = [['tanh','tanh'],['tanh','tanh'],['tanh','tanh']]    # Activation functions
    
training_parameters["batch_size"] = 512         # Batch size for the gradient descent
training_parameters["initial_learning_rate"] = 1.0e-3           # Initial learnign rate
training_parameters["decay_steps"] = 50         # Exponential decay: done each decay_steps
training_parameters["decay_rate"] = 0.92        # Exponential decay: rate
training_parameters["staircase"] = True         # Stair case or continuous

training_parameters["alpha_reg"] = 0.0    # L2 regularization coefficient
        
training_parameters["epochs_list"] = [1,1,1]    # number of epochs for each cluster


training_parameters["hard_constraints_model"] = 0   # Hard constraint: 0=no constraints, 1=constraints on atomic balance


# Models instantiation
mlp_model = MLPModel(training_parameters)

# Models training
mlp_model.train_models()


