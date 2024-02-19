from ai_reacting_flows.ann_model_generation.MLP_model_pytorch import *

config = {
    ###CONFIG COMMUNS###
    'communs' : {
        'folder' : "./MODEL_Learning_Tensorflow", 
        'Framework' :'Pytorch', # Tensorflow or Pytorch 
        'nb_units_in_layers_list' : [[40,40],[80,80]] , 
        'layers_activation_list' : [['tanh','tanh'],['tanh', 'tanh']], 
        'layers_type' : "dense",
        'dataset_path' : "./STOCH_DTB_H2_HOTSPOT_MK" , 
        'dt_simu' : 5.0e-07,
        'fuel' : "H2",
        'mechanism_type' : "detailed",
        'device' : "cuda",
        'remove_N2' : True,
        'hard_constraints_model' : 0,
	    'log_X' : 1, 
	    'log_Y' : 1,
        
    }, 
    
    'training' :{
        'new_model' : True,
        'batch_size' : 512,
        'shuffle' : False,
        'loss_function' : "mean_squared_error",
        'epoch' : [1000,1000], #,500,500,500,500] , 
        'optimizer' : "adam",
        'initial_learning_rate' : 1.0e-3,
        'decay_steps' : 50, 
        'decay_rate' : 0.92,     



    },

    'testing' : {
        'spec_to_plot' : ["H2", "O2", "N2"],
        'pv_species' : ["H2O"], 
        'yk_renormalization' : False, 
        'hybrid_ann_cvode' : False , 
        'hybrid_ann_cvode_tol' : 5.0e-05, 
        'output_omegas' : True, 
        'threshold' : 1e-10,
        'phi' : 0.4,
        'T0' : 1200,
        'pressure' : 101325.0,
        'dt' : 0.5e-6,
        'nb_ite' : 1,
        


    },


}

#Initialisation Models and loading
Test = ClusterModels(config['communs'])
Test.init_testing(config['testing'])



phi = 0.4
T0 = 1200.0
pressure = 101325.0
dt = 0.5e-6
nb_ite = 500

Test.Test_0D_ignition_Pytorch(phi, T0, pressure, dt, nb_ite)
#Test.Test_0D_ignition_Pytorch(**config['testing'])

phi = 0.4
T0 = 300.0
pressure = 101325.0
dt = 0.5e-6
T_threshold = 600.0

Test.Test_1D_ignintion_Pytorch(phi, T0, pressure, dt, T_threshold)