from ai_reacting_flows.ann_model_generation.MLP_model_pytorch import *

config = {
    ###CONFIG COMMUNS###
    'communs' : {
        'folder' : "./LEARNING_PYTORCH_H2_HOTSPOT_TEST3", 
        'nb_units_in_layers_list' : [[40,40],[80,80]] , 
        'layers_activation_list' : [['tanh','tanh'],['tanh', 'tanh']], 
        'layers_type' : "dense",
        'dataset_path' : "/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows_pytorch/scripts/STOCH_DTB_H2_HOTSPOT_MK/dtb" , 
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
        'phi' : 0.4,
        'T0' : 1200,
        'pressure' : 101325.0,
        'dt' : 0.5e-6,
        'nb_ite' : 500,
        


    },


}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
if use_cuda :
    print('___CUDNN VERSION:',torch.backends.cudnn.version())
    print('___Number CUDA devices:', torch.cuda.device_count())
    print('___Cuda device name:',torch.cuda.get_device_name(0))
    print('___CUDA device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

#if not working in terminal : unset LD_LIBRARY_PATH 

# Models instantiation
model = ClusterModels(config['communs'])
model.init_training(config['training'])

# Models training: all clusters
model.train_all_models()
