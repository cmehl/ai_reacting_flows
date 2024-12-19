import os
# from re import S
# from sqlite3 import enable_callback_tracebacks
import sys

import matplotlib.pyplot as plt

import shelve
import shutil

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler

import joblib

import tensorflow as tf
import keras._tf_keras.keras.backend as K
from keras._tf_keras.keras import models
from keras._tf_keras.keras import layers
from keras._tf_keras.keras import optimizers
from keras._tf_keras.keras import losses
from keras._tf_keras.keras import metrics
#from keras._tf_keras.keras import callbacks
from keras._tf_keras.keras import regularizers
from keras._tf_keras.keras import initializers
# from keras._tf_keras.keras.utils import plot_model

from ai_reacting_flows.ann_model_generation.tensorflow_custom import sum_species_metric
# from ai_reacting_flows.ann_model_generation.tensorflow_custom import AtomicConservation, AtomicConservation_RR, AtomicConservation_RR_lsq
from ai_reacting_flows.ann_model_generation.tensorflow_custom import GetN2Layer, ZerosLayer, GetLeftPartLayer, GetRightPartLayer
from ai_reacting_flows.ann_model_generation.tensorflow_custom import ResidualBlock

import ai_reacting_flows.tools.utilities as utils

import cantera as ct

sns.set_style("dark")
# Using float64
tf.keras.backend.set_floatx('float64')

tf.random.set_seed(2022)

class MLPModel(object):

    def __init__(self, training_parameters):
        
        # Model folder name
        self.model_name = "MODEL_" + training_parameters["model_name_suffix"]
        
        # Dataset folder name
        self.dataset_path = training_parameters["dataset_path"]

        # New model or retrain models in an existing folder
        self.new_model_folder = training_parameters["new_model_folder"]
        
        # Simulation time-step
        self.dt_simu = training_parameters["dt_simu"]

        # Fuel
        self.fuel = training_parameters["fuel"]
        self.mechanism_type = training_parameters["mechanism_type"]

        # Remove N2
        self.remove_N2 = training_parameters["remove_N2"]

        # Network shapes
        self.nb_units_in_layers_list = training_parameters["nb_units_in_layers_list"]
        self.layers_activation_list = training_parameters["layers_activation_list"]
        self.layers_type = training_parameters["layers_type"]
    
        # Optimization parameters
        self.batch_size = training_parameters["batch_size"]
        self.initial_learning_rate = training_parameters["initial_learning_rate"]
        # Parameters of the exponential decay schedule (learning rate decay)
        self.decay_steps = training_parameters["decay_steps"]
        self.decay_rate = training_parameters["decay_rate"]
        self.staircase = training_parameters["staircase"]

        # L2 regularization coefficient
        self.alpha_reg = training_parameters["alpha_reg"] 
        
        # epochs number
        self.epochs_list = training_parameters["epochs_list"]

        # Enforcing hard contraints
        # 0: no hard constraints; 1: atomic masses
        self.hard_constraints_model = training_parameters["hard_constraints_model"]

        # Box-Cox parameter (set by default to 0.1)
        self.lambda_bct = 0.1

        # Load databased-linked parameter using shelve file
        shelfFile = shelve.open(self.dataset_path + "/dtb_params")
        self.threshold = shelfFile["threshold"]
        self.log_transform_X = shelfFile["log_transform_X"]
        self.log_transform_Y = shelfFile["log_transform_Y"]
        self.output_omegas = shelfFile["output_omegas"]
        self.clustering_type = shelfFile["clusterization_method"]
        self.with_N_chemistry = shelfFile["with_N_chemistry"]
        shelfFile.close()

        # Model's path
        self.directory = "./" + self.model_name
        if self.new_model_folder:
            print(">> A new folder is created.")
            # Remove folder if already exists
            shutil.rmtree(self.directory, ignore_errors=True)
            # Create folder
            os.makedirs(self.directory)
        else:
            if not os.path.exists(self.directory):
                sys.exit(f'ERROR: new_model_folder is set to False but model {self.directory} does not exist')
            print(f">> Existing model folder {self.directory} is used. \n")


        # Parameters which are to be kept constant if new_model_folder is False
        if self.new_model_folder:  # Store parameters
            shelfFile = shelve.open(self.directory + "/restart_params")
            shelfFile["dataset_path"] = self.dataset_path
            shelfFile["dt_simu"] = self.dt_simu
            shelfFile["fuel"] = self.fuel
            shelfFile["mechanism_type"] = self.mechanism_type
            shelfFile.close()
        else:   # read parameters
            shelfFile = shelve.open(self.directory + "/restart_params")
            self.dataset_path = shelfFile["dataset_path"]
            self.dt_simu = shelfFile["dt_simu"]
            self.fuel = shelfFile["fuel"]
            self.mechanism_type = shelfFile["mechanism_type"]
            shelfFile.close()
            #
            print("RE-READ PARAMETERS:")
            print(f">> The used dataset is: {self.dataset_path}")
            print(f">> The used time-step is: {self.dt_simu}")
            print(f">> The considered fuel is: {self.fuel}")
            print(f">> The chemical mechanism type is: {self.mechanism_type} \n")

        # Checking consistency of inputs
        self.check_inputs()

        # Get the number of clusters
        self.nb_clusters = len(next(os.walk(self.dataset_path))[1])
        print("CLUSTERING:")
        print(f">> Number of clusters is: {self.nb_clusters}")
        
        # Create __init__.py for later use of python files
        if self.new_model_folder:
            with open(self.directory + "/__init__.py", 'w'): pass

        # Training stats
        if self.new_model_folder:
            os.mkdir(self.directory + "/training")
            os.mkdir(self.directory + "/training/training_curves")
            os.mkdir(self.directory + "/evaluation" )

            # Adding copies of clustering parameters for later use in inference
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

        # Defining mechanism file (either detailed or reduced)
        if self.mechanism_type=="detailed":
            self.mechanism = self.dataset_path + "/mech_detailed.yaml"
        elif self.mechanism_type=="reduced":
            self.mechanism = self.dataset_path + "/mech_reduced.yaml"

        # We copy the mechanism files in order to use them for testing
        if self.new_model_folder:
            if self.mechanism_type=="detailed":
                shutil.copy(self.dataset_path + "/mech_detailed.yaml", self.directory)
            elif self.mechanism_type=="reduced":
                shutil.copy(self.dataset_path + "/mech_reduced.yaml", self.directory)


        # Saving parameters using shelve file for later use in testing
        shelfFile = shelve.open(self.directory + '/model_params')
        #
        shelfFile["threshold"] = self.threshold
        shelfFile["log_transform_X"] = self.log_transform_X
        shelfFile["log_transform_Y"] = self.log_transform_Y
        shelfFile["output_omegas"] = self.output_omegas
        shelfFile["hard_constraints_model"] = self.hard_constraints_model
        shelfFile["remove_N2"] = self.remove_N2
        shelfFile["mechanism_type"] = self.mechanism_type
        #
        shelfFile.close()


        # PREPROCESSING 
        # If hard constraints are imposed, the conservation matrix is built
        if self.hard_constraints_model>0:
            self.build_conservation_matrix()

        # Dictionary for solving models
        self.models_dict = {}

    def get_data(self, i_cluster):
        
        X_train = pd.read_csv(filepath_or_buffer= self.dataset_path + f"/cluster{i_cluster}/X_train.csv")
        Y_train = pd.read_csv(filepath_or_buffer= self.dataset_path + f"/cluster{i_cluster}/Y_train.csv")
            
        X_val = pd.read_csv(filepath_or_buffer= self.dataset_path + f"/cluster{i_cluster}/X_val.csv")
        Y_val = pd.read_csv(filepath_or_buffer= self.dataset_path + f"/cluster{i_cluster}/Y_val.csv")

        return X_train, X_val, Y_train, Y_val

    #------------------------------------------------------------------------------------
    # MODEL TRAINING FUNCTIONS
    #------------------------------------------------------------------------------------

    def train_model_cluster_i(self, i_cluster):

        if i_cluster >= self.nb_clusters:
            sys.exit(f"The cluster identifier {i_cluster} is higher than the number of clusters !")

        # Using mechanism set as input to get species name and number of species
        # /!\ We assume that this is consistent with database /!\
        gas = ct.Solution(self.mechanism)
        spec_names = gas.species_names
        nb_spec = gas.n_species

        # If we do not have the N chemistry (i.e. N2 is constant), we can either remove N2 or force it constant
        if self.with_N_chemistry is False:
            if self.remove_N2:
                spec_names.remove("N2")
                nb_spec = nb_spec - 1

            else:
                # N2 index
                self._n2_index = spec_names.index("N2")

                # Number of species on left and right of N2 in list
                self._left_n2 = len(spec_names[:self._n2_index])
                self._right_n2 = len(spec_names[self._n2_index+1:])

        print(50*"-")
        print(50*"-")
        print(f"                BUILDING MODEL FOR CLUSTER {i_cluster}")
        print(50*"-")
        print(50*"-"+"\n")

        # ===============================================================================================================
        #                                             CLUSTER SPECIFIC PARAMETERS
        # ===============================================================================================================
        
        # Network shapes
        nb_units_in_layers = self.nb_units_in_layers_list[i_cluster]
        layers_activation = self.layers_activation_list[i_cluster]
        
        # Number of epochs
        epochs = self.epochs_list[i_cluster]

        # ===============================================================================================================
        #                                             GETTING DATA
        # ===============================================================================================================

        # Getting data
        X_train, X_val, Y_train, Y_val = self.get_data(i_cluster)

        if self.remove_N2:
            X_train = X_train.drop("N2_X", axis=1)
            X_val = X_val.drop("N2_X", axis=1)
            Y_train = Y_train.drop("N2_Y", axis=1)
            Y_val = Y_val.drop("N2_Y", axis=1)
        
        Y_cols = Y_train.columns
        X_cols = X_train.columns

        
        # Verifying species names conformity
        nb_spec = len(Y_cols)
        spec_names_dtb = []
        for k in range(nb_spec):
            spec_names_dtb.append(Y_cols[k].split('_')[0])

        try:
            spec_names==spec_names_dtb
        except:
            sys.exit("Error: mechanism species names do not correspond to species in database !")
            
        
        # Define model's targets
        targets = spec_names.copy()
        
        print(f" >> Number of training samples: {X_train.shape[0]}")    
        print(f" >> Number of validation samples: {X_val.shape[0]} \n")       
        
        # ================================================================================================================
        #                                         NORMALIZING INPUT/OUTPUT
        # ===============================================================================================================

        # QUESTION: SHOULD WE USE SAME SCALER FOR Y THAN FOR X ?  -> TO TEST

        # NORMALIZING X
        Xscaler = StandardScaler()
        # Fit scaler
        Xscaler.fit(X_train)
        # Transform data (remark: automatically transform to numpy array)
        X_train = Xscaler.transform(X_train)
        
        X_val = Xscaler.transform(X_val)
        # X Scaling parameters 
        self._param1_X, self._param2_X = self.get_scaler_params(Xscaler)
        
        # NORMALIZING Y
        # Choose scaler
        Yscaler = StandardScaler()

        # Fit scaler
        Yscaler.fit(Y_train)
    
        # Transform data (remark: automatically transform to numpy array)
        Y_train = Yscaler.transform(Y_train)
        Y_val = Yscaler.transform(Y_val)
                
        #TODO: CHECK THAT THIS IS DONE IN DATABASE GENERATION
        # Is this necessary ??
        # if self.output_omegas==True and self.log_transform==True:
        #     Y_train[:,spec_names.index("N2")] = 0.0
        #     Y_val[:,spec_names.index("N2")] = 0.0
        
        # Y Scaling parameters
        self._param1_Y, self._param2_Y = self.get_scaler_params(Yscaler)
            
        # Converting numpy arrays into keras tensors for use in custom losses & metrics
        param1_Y_tensor = K.variable(self._param1_Y, dtype="float64")
        param2_Y_tensor = K.variable(self._param2_Y, dtype="float64")
        
        # Saving scalers
        joblib.dump(Xscaler, self.directory + f'/Xscaler_cluster{i_cluster}.pkl')
        joblib.dump(Yscaler, self.directory + f'/Yscaler_cluster{i_cluster}.pkl')
            
        # Saving mean and variance in matrix form to be read by CONVERGE
        np.savetxt(self.directory + f'/norm_param_X_cluster{i_cluster}.dat', np.vstack([Xscaler.mean_, Xscaler.var_]).T)
        np.savetxt(self.directory + f'/norm_param_Y_cluster{i_cluster}.dat', np.vstack([Yscaler.mean_, Yscaler.var_]).T)

        # If reaction rates as outputs, we have to deal with non reacting species
        # To do that, we define other parameyers for unscaling (to avoid "(0-0)/0")
        if self.output_omegas:
            self._param2_Y_scale = np.copy(self._param2_Y)
            for i in range(self._param2_Y.shape[0]):
                
                if self._param2_Y[i]==0.0:
                    self._param2_Y_scale[i] = np.infty

        # =================================================================================================================
        #                                               ANN LEARNING
        # =================================================================================================================
        
        print(50*"-")
        print("                MODEL TRAINING")
        print(50*"-"+"\n")
        
        # Using float64
        tf.keras.backend.set_floatx('float64')
    
        # Model generation
        if self.with_N_chemistry:
            model = self.generate_nn_model(X_train.shape[1], Y_train.shape[1], nb_units_in_layers, layers_activation)
        else:
            if self.remove_N2:
                model = self.generate_nn_model(X_train.shape[1], Y_train.shape[1], nb_units_in_layers, layers_activation)
            else:
                model = self.generate_nn_model_N2_cte(X_train.shape[1], Y_train.shape[1], nb_units_in_layers, layers_activation)    

        #========================================== defining the optimizer ======================================
        #======================================================================================================== 
        
        # Build the learning rate schedule 
        lr_schedule = optimizers.schedules.ExponentialDecay(
                                    initial_learning_rate=self.initial_learning_rate,
                                    decay_steps=self.decay_steps,
                                    decay_rate=self.decay_rate,
                                    staircase=self.staircase)
    
        # Build the optimizer
        optimizer = optimizers.Adam()

        # empty list for keras callbacks
        callbacks_list=[] 

        # Callback: LR scheduler
        callbacks_list.append([tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)])

        
        # Metrics
        metrics_list=[metrics.mape,metrics.mae,metrics.mse]
        metrics_list.append(sum_species_metric(param1_Y_tensor, param2_Y_tensor, self.log_transform_Y))            
        
        # define the loss function   
        loss=losses.mean_squared_error
        # loss=losses.mean_absolute_error
            
        # compile the model
        model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=metrics_list)
            
        # fit the model
        history = model.fit(X_train,
                        Y_train,
                        validation_data=(X_val,Y_val),
                        epochs=epochs,
                        batch_size=self.batch_size,
                        validation_freq=1,
                        callbacks=callbacks_list,
                        verbose=1)
        
        
        # Save the weights
        model.save_weights(self.directory +  f'/model_weights_cluster{i_cluster}.h5')
        
        # Save the model architecture
        with open(self.directory + f'/model_architecture_cluster{i_cluster}.json', 'w') as f:
            f.write(model.to_json())
            
        # Also save the model in SavedModel format
        if os.path.exists(self.directory + f'/my_model_cluster{i_cluster}'):
            shutil.rmtree(self.directory + f'/my_model_cluster{i_cluster}', ignore_errors=True)
        model.save(self.directory + f'/my_model_cluster{i_cluster}')
        
        # Saving a representation of the model
        # plot_model(model, to_file=self.directory+ f'/model_plot{i_cluster}.png', show_shapes=True, show_layer_names=True)

        # Storing model
        self.models_dict[i_cluster] = model
            
        #%%
        # =====================================================================================================================
        #                                               PLOTTING
        # ====================================================================================================================
        
        self.history_dict = history.history
        
        # defining the metrics to plot 
        targets = spec_names.copy()
        targets[targets=='Temperature']='T'
        
        targets.remove('T')
        
        conservation_metrics = ['sum_species']
        
        # ====================== plot loss curves ===========
        
        fig,ax=plt.subplots()
        ax.set_yscale('log')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(f"Cluster {i_cluster}", fontsize=15)
        plt.plot(self.history_dict['loss'])
        plt.plot(self.history_dict['val_loss'])
        plt.legend(['train','validation'])
        plt.savefig( self.directory + f"/training/training_curves/loss_cluster{i_cluster}.png") 
        plt.show()
        

        #TODO: ONLY SUM_PECIES
        for metric in conservation_metrics:
            
            fig,ax=plt.subplots()
            ax.set_yscale('log')
            
            plt.xlabel('epoch')
            plt.ylabel(metric)
            plt.title(f"Cluster {i_cluster}", fontsize=15)
            plt.plot(self.history_dict[metric])
            plt.plot(self.history_dict['val_'+metric])
            plt.legend(['train','validation'])
            plt.savefig(self.directory  + "/training/training_curves/" + metric + f'_cluster{i_cluster}.png')        
            plt.show()
        
        # =====================================================================================================================
        #                                      ERROR ANALYSIS
        # ====================================================================================================================
        
        # ============== save global model errors on train/validation datasets
        
        evaluation_metrics = [ metric for metric in self.history_dict.keys() if ('val' not in metric ) and (metric!='lr') ]
        columns = ['dataset'] + evaluation_metrics
        model_results = pd.DataFrame(index=range(3), columns=columns)
        model_results.iloc[0] = ['train'] + model.evaluate(X_train,Y_train)
        model_results.iloc[1] = ['valid'] + model.evaluate(X_val,Y_val)
        
        
        model_results.to_csv( self.directory  + f"/evaluation/errors_cluster{i_cluster}.csv" ,sep=';',index=False)
        
        # save history 
        pd.DataFrame(self.history_dict).to_csv(self.directory + f"/training/training_curves/history_cluster{i_cluster}.csv" ,sep=';',index=False)
        
        # ============== in depth error analysis
        
        # ----Validation data
        if self.log_transform_Y>0:
            log_Y_val_pred = model.predict(X_val)
            #
            Y_val_unscaled = Yscaler.inverse_transform(Y_val)

            if self.log_transform_Y==1: # LOG
                Y_val_unscaled = np.exp(Y_val_unscaled)
            elif self.log_transform_Y==2: # BCT
                Y_val_unscaled = (self.lambda_bct*Y_val_unscaled+1.0)**(1./self.lambda_bct)
            #
            Y_val_pred_unscaled = Yscaler.inverse_transform(log_Y_val_pred)

            
            if self.log_transform_Y==1: # LOG
                Y_val_pred_unscaled = np.exp(Y_val_pred_unscaled)
            if self.log_transform_Y==2: # BCT
                Y_val_pred_unscaled = (self.lambda_bct*Y_val_pred_unscaled+1.0)**(1./self.lambda_bct)
                
        else:
            Y_val_pred = model.predict(X_val)
            #
            Y_val_unscaled = Yscaler.inverse_transform(Y_val)
            #
            Y_val_pred_unscaled = Yscaler.inverse_transform(Y_val_pred)

        #
        if self.log_transform_X==1: # LOG
            X_val_unscaled = Xscaler.inverse_transform(X_val)
            X_val_unscaled[:,1:] = np.exp(X_val_unscaled[:,1:])
        elif self.log_transform_X==2: # BCT
            X_val_unscaled = Xscaler.inverse_transform(X_val)
            X_val_unscaled[:,1:] = (self.lambda_bct*X_val_unscaled[:,1:]+1.0)**(1./self.lambda_bct) 
        else:
            X_val_unscaled = Xscaler.inverse_transform(X_val)
            
        # Prediction error (in %)
        # errors_pred_val = 100.0*np.divide(np.absolute(Y_val_pred_unscaled - Y_val_unscaled), Y_val_unscaled)
        errors_pred_val = np.absolute(Y_val_pred_unscaled - Y_val_unscaled)
            
        # Storing in csv
        data_array = np.concatenate((X_val_unscaled, Y_val_unscaled, Y_val_pred_unscaled, errors_pred_val), axis=1)
        error_cols = [str(col) + '_err' for col in Y_cols]
        Y_pred_cols = [str(col) + '_pred' for col in Y_cols]
        columns = list(X_cols) + list(Y_cols) + Y_pred_cols + error_cols
        dtb_val = pd.DataFrame(data_array, columns = columns)
        dtb_val.to_csv( self.directory  + f"/evaluation/validation_predictions_cluster{i_cluster}.csv" ,sep=';',index=False)
            

        # Plot function template (scatter plots)
        # sns.relplot(x="CH4_X", y="CH4_Y_err", data=dtb_val)

    def train_model_all_clusters(self):

        # Loop on clusters
        for i_cluster in range(self.nb_clusters):
            self.train_model_cluster_i(i_cluster)

    #------------------------------------------------------------------------------------
    # MODEL DEFINITION FUNCTIONS
    #------------------------------------------------------------------------------------

    def generate_nn_model_N2_cte(self, n_X, n_Y, nb_units_in_layers, layers_activation):

        # Choosing hidden layer type: dense or resblock
        if self.layers_type=="dense":
            hidden_layer = layers.Dense
            prefix_layer = "dense_layer"
        elif self.layers_type=="resnet":
            hidden_layer = ResidualBlock
            prefix_layer = "resblock_layer"

        # Layer number (might differ from i in the case of resnet -> if there is a dense layer we want first resnet to be called resblock_layer_2 for easier reading in C++)
        layer_nb = 1

        layers_dict = {}
        
        layers_dict["input_layer"] = layers.Input(shape=(n_X,), name="input_layer")

        # We dissociate N2 from other species. 
        # We assume that we deal with air combustion and no N2 reactions. 
        # /!\ If this is not the case, this part of the code must be change /!\
        n2_layer = GetN2Layer(1, self._n2_index+1, kernel_initializer=initializers.GlorotUniform())(layers_dict["input_layer"])

        # We need a tensor always equal to zero for N2 reaction rate output
        if self.output_omegas==True:
            zero_layer = ZerosLayer(1, kernel_initializer=initializers.GlorotUniform())(n2_layer)

        # Getting split part of inputs vector (part on left and right of N2)
        yk_layer_1 = GetLeftPartLayer(self._left_n2, self._n2_index+1, kernel_initializer=initializers.GlorotUniform())(layers_dict["input_layer"])
        yk_layer_2 = GetRightPartLayer(self._right_n2, self._n2_index+2, kernel_initializer=initializers.GlorotUniform())(layers_dict["input_layer"])
        yk_layer = layers.Concatenate(axis=1)([yk_layer_1, yk_layer_2])

        if self.layers_type=="resnet":  
            layers_dict["dense_layer_1"] = layers.Dense(units=nb_units_in_layers[0],kernel_regularizer=regularizers.l2(self.alpha_reg),
                                            activation=layers_activation[0], kernel_initializer=initializers.GlorotUniform(), name=f"dense_layer_{layer_nb}")(yk_layer)
            layer_nb += 1

            layers_dict["hidden_layer_1"] = hidden_layer(units=nb_units_in_layers[0],kernel_regularizer=regularizers.l2(self.alpha_reg),
                                            activation=layers_activation[0], kernel_initializer=initializers.GlorotUniform(), name=prefix_layer + f"_{layer_nb}")(layers_dict["dense_layer_1"])
            layer_nb += 1
        else:
            layers_dict["hidden_layer_1"] = hidden_layer(units=nb_units_in_layers[0],kernel_regularizer=regularizers.l2(self.alpha_reg),
                                            activation=layers_activation[0], kernel_initializer=initializers.GlorotUniform(), name=prefix_layer + f"_{layer_nb}")(yk_layer)
            layer_nb += 1

        for i in range(2, len(nb_units_in_layers)+1):
            layers_dict[f"hidden_layer_{i}"] = hidden_layer(units=nb_units_in_layers[i-1],kernel_regularizer=regularizers.l2(self.alpha_reg),
                                                            activation=layers_activation[i-1] ,kernel_initializer=initializers.GlorotUniform(), name=prefix_layer + f"_{{layer_nb}}")(layers_dict[f"hidden_layer_{i-1}"])
            layer_nb += 1
            
            #=========================== model's output definition ( constrained or not )=====================================
            
        # if self.hard_constraints_model==0:

        # layers_dict['output_layer'] = layers.Dense(units=Y_train.shape[1], kernel_regularizer=regularizers.l2(alpha_reg), kernel_initializer=initializers.GlorotUniform(),name='output_layer')(layers_dict[f"activation_layer_{len(nb_units_in_layers)}"])
        output_layer = layers.Dense(units=n_Y-1, kernel_regularizer=regularizers.l2(self.alpha_reg), kernel_initializer=initializers.GlorotUniform(),name='output_layer')(layers_dict[f"hidden_layer_{len(nb_units_in_layers)}"]) 

        # We recreate whole vector with correct ordering 
        output_layer_1 = GetLeftPartLayer(self._left_n2, self._n2_index, kernel_initializer=initializers.GlorotUniform())(output_layer)
        output_layer_2 = GetRightPartLayer(self._right_n2, self._n2_index, kernel_initializer=initializers.GlorotUniform())(output_layer)
        if self.output_omegas==True: 
            layers_dict['output_layer'] = layers.Concatenate(axis=1)([output_layer_1, zero_layer,output_layer_2])
        else:
            layers_dict['output_layer'] = layers.Concatenate(axis=1)([output_layer_1, n2_layer, output_layer_2])

        # elif self.hard_constraints_model>0:
                
        #     # Intermediate dense layer
        #     # layers_dict["output_layer_interm"] = layers.Dense(units=Y_train.shape[1],kernel_regularizer=regularizers.l2(alpha_reg),kernel_initializer=initializers.GlorotUniform(), name="output_layer_interm")(layers_dict[f"activation_layer_{len(nb_units_in_layers)}"])
                
        #     output_layer_interm = layers.Dense(units=n_Y-1, kernel_regularizer=regularizers.l2(self.alpha_reg), kernel_initializer=initializers.GlorotUniform(),name='output_layer_interm')(layers_dict[f"hidden_layer_{len(nb_units_in_layers)}"]) 


        #     output_layer_1 = GetLeftPartLayer(self._left_n2, self._n2_index, kernel_initializer=initializers.GlorotUniform())(output_layer_interm)
        #     output_layer_2 = GetRightPartLayer(self._right_n2, self._n2_index, kernel_initializer=initializers.GlorotUniform())(output_layer_interm)
                
        #     if self.output_omegas==True: 
        #         layers_dict['output_layer_interm'] = layers.Concatenate(axis=1)([output_layer_1, zero_layer, output_layer_2])
        #     else:
        #         layers_dict['output_layer_interm'] = layers.Concatenate(axis=1)([output_layer_1, n2_layer, output_layer_2])
                

        #     if self.hard_constraints_model==1:
        #         # Layer enforcing physical constraint
        #         # Atomic conservation
        #         if self.output_omegas==True:
        #             layers_dict["output_layer"] = AtomicConservation_RR(n_Y, self._param1_Y, self._param2_Y, self._param2_Y_scale, 
        #                                                                     self.A_atomic_t, self.A_inv_final_t,kernel_initializer=initializers.GlorotUniform())(layers_dict["output_layer_interm"])
        #         else:
        #             layers_dict["output_layer"] = AtomicConservation(n_Y, self._param1_X, self._param2_X, self._param1_Y, 
        #                                                     self._param2_Y, self.log_transform_Y, self.threshold, self.A_atomic_t, self.A_inv_final_t,kernel_initializer=initializers.GlorotUniform())([layers_dict["output_layer_interm"],layers_dict["input_layer"]])
                                
        #     elif self.hard_constraints_model==2:

        #         # Layer enforcing physical constraint
        #         # Atomic conservation
        #         layers_dict["output_layer"] = AtomicConservation_RR_lsq(n_Y, self._param1_Y, self._param2_Y, self._param2_Y_scale, 
        #                                                                 self.L, kernel_initializer=initializers.GlorotUniform())(layers_dict["output_layer_interm"])
        
        # ================                Define model   ==========================================================
        
        model = models.Model(layers_dict["input_layer"], outputs=layers_dict["output_layer"], name="main_model")

        return model

    def generate_nn_model(self, n_X, n_Y, nb_units_in_layers, layers_activation):

        # Choosing hidden layer type: dense or resblock
        if self.layers_type=="dense":
            hidden_layer = layers.Dense
            prefix_layer = "dense_layer"
        elif self.layers_type=="resnet":
            hidden_layer = ResidualBlock
            prefix_layer = "resblock_layer"


        layers_dict = {}
        
        layers_dict["input_layer"] = layers.Input(shape=(n_X,), name="input_layer")

        # Layer number (might differ from i in the case of resnet -> if there is a dense layer we want first resnet to be called resblock_layer_2 for easier reading in C++)
        layer_nb = 1

        # If resnet, we need an input to the resblock with same dimension as the units (to be able to perform the sum inputs+outputs)
        if self.layers_type=="resnet":

            layers_dict["dense_layer_1"] = layers.Dense(units=nb_units_in_layers[0],kernel_regularizer=regularizers.l2(self.alpha_reg),
                                        activation=layers_activation[0], kernel_initializer=initializers.GlorotUniform(), name= f"dense_layer_{layer_nb}")(layers_dict["input_layer"])
            layer_nb += 1

            layers_dict["hidden_layer_1"] = hidden_layer(units=nb_units_in_layers[0],kernel_regularizer=regularizers.l2(self.alpha_reg),
                                        activation=layers_activation[0], kernel_initializer=initializers.GlorotUniform(), name=prefix_layer + f"_{layer_nb}")(layers_dict["dense_layer_1"])
            layer_nb += 1
        else:
            layers_dict["hidden_layer_1"] = hidden_layer(units=nb_units_in_layers[0],kernel_regularizer=regularizers.l2(self.alpha_reg),
                                            activation=layers_activation[0], kernel_initializer=initializers.GlorotUniform(), name=prefix_layer + f"_{layer_nb}")(layers_dict["input_layer"])
            layer_nb += 1
            
        for i in range(2, len(nb_units_in_layers)+1):
            layers_dict[f"hidden_layer_{i}"] = hidden_layer(units=nb_units_in_layers[i-1],kernel_regularizer=regularizers.l2(self.alpha_reg),
                                                            activation=layers_activation[i-1] ,kernel_initializer=initializers.GlorotUniform(), name=prefix_layer + f"_{layer_nb}")(layers_dict[f"hidden_layer_{i-1}"])
            layer_nb += 1
         
        #=========================== model's output definition ( constrained or not )=====================================
            
        # if self.hard_constraints_model==0:
        layers_dict['output_layer'] = layers.Dense(units=n_Y, kernel_regularizer=regularizers.l2(self.alpha_reg), kernel_initializer=initializers.GlorotUniform(),name='output_layer')(layers_dict[f"hidden_layer_{len(nb_units_in_layers)}"]) 

        # elif self.hard_constraints_model>0:
                
        #     # Intermediate dense layer   
        #     layers_dict['output_layer_interm'] = layers.Dense(units=n_Y, kernel_regularizer=regularizers.l2(self.alpha_reg), kernel_initializer=initializers.GlorotUniform(),name='output_layer_interm')(layers_dict[f"hidden_layer_{len(nb_units_in_layers)}"])

        #     if self.hard_constraints_model==1:
        #         # Layer enforcing physical constraint
        #         # Atomic conservation
        #         if self.output_omegas==True:
        #             layers_dict["output_layer"] = AtomicConservation_RR(n_Y, self._param1_Y, self._param2_Y, self._param2_Y_scale, 
        #                                                                     self.A_atomic_t, self.A_inv_final_t,kernel_initializer=initializers.GlorotUniform())(layers_dict["output_layer_interm"])
        #         else:
        #             layers_dict["output_layer"] = AtomicConservation(n_Y, self._param1_X, self._param2_X, self._param1_Y, 
        #                                                     self._param2_Y, self.log_transform_Y, self.threshold, self.A_atomic_t, self.A_inv_final_t,kernel_initializer=initializers.GlorotUniform())([layers_dict["output_layer_interm"],layers_dict["input_layer"]])
                                
        #     elif self.hard_constraints_model==2:

        #         # Layer enforcing physical constraint
        #         # Atomic conservation
        #         layers_dict["output_layer"] = AtomicConservation_RR_lsq(n_Y, self._param1_Y, self._param2_Y, self._param2_Y_scale, 
        #                                                                 self.L, kernel_initializer=initializers.GlorotUniform())(layers_dict["output_layer_interm"])
        
        # ================                Define model   ==========================================================
        
        model = models.Model(layers_dict["input_layer"], outputs=layers_dict["output_layer"], name="main_model")

        return model

    #------------------------------------------------------------------------------------
    # MISC FUNCTIONS
    #------------------------------------------------------------------------------------

    def build_conservation_matrix(self):

        # Using mechanism set as input to get species name and number of species
        # /!\ We assume that this is consistent with database /!\
        gas = ct.Solution(self.mechanism)
        spec_names = gas.species_names

        # If N2 not considered we remove it
        if self.remove_N2:
            spec_names.remove("N2")

        nb_spec = len(spec_names)

        # Matrix with number of each atom in species (order of rows: C, H, O, N)
        atomic_array = utils.parse_species_names(spec_names)
        
        # Atomic mass per elements (dictionary)
        #mass_per_atom = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007}
        mass_per_atom_array = np.array([12.011, 1.008, 15.999, 14.007])  # Having the array is also convenient
        
        # Atomic mass per species (numpy array)
        mol_weights = utils.get_molecular_weights(spec_names)
        #mol_weights_tensor = K.variable(mol_weights, dtype="float64")
        
        # Matrix for computing atomic conservation
        A_atomic = np.copy(atomic_array)
        for j in range(4):
            A_atomic[j,:] *=  mass_per_atom_array[j]
        for k in range(nb_spec):
            A_atomic[:,k] /=  mol_weights[k]

        # If hydrogen we discard carbon, so that the writing in keras custom layer is more simple
        if self.fuel=="H2":
            A_atomic = A_atomic[1:,:]

        # If we remove N2, we assume that we have no nitrogen at all
        if self.remove_N2==True:
            A_atomic = A_atomic[:-1,:]

        # We will need the transpose
        self.A_atomic_t = np.transpose(A_atomic)
            
        
        # For atomic conservation, we need to build the inverse of the following matrix
        # A=(ajk) where ajk = (Mj/Mk)*nkj
        # Where it is done for a subset of species used to balance conservation
        if self.hard_constraints_model==1:

            # We tackle each case: whether or not H and N are included
            if self.fuel=="H2":
                if self.remove_N2==True: # N chemistry is not kept
                    balancing_species = ["H2O", "O2"]
                    mass_per_atom_array = np.array([1.008, 15.999])
                else:
                    balancing_species = ["H2O", "O2", "N2"]
                    mass_per_atom_array = np.array([1.008, 15.999, 14.007])
            else:
                if self.remove_N2==True: # N chemistry is not kept
                    balancing_species = ["CO2", "H2O", "O2"]
                    mass_per_atom_array = np.array([12.011, 1.008, 15.999, 14.007])
                else:
                    balancing_species = ["CO2", "H2O", "O2", "N2"]
                    mass_per_atom_array = np.array([12.011, 1.008, 15.999])
                
            # We take atomic_array and just keep desired species
            A_reduced = A_atomic[:,[spec_names.index(spec) for spec in balancing_species]]
                    
            # Inverting matrix
            try:
                A_reduced_inv = np.linalg.inv(A_reduced)
            except np.linalg.LinAlgError:
                sys.exit("Error: the matrix of atomic constraints balancing species is not invertible !")
                
            # After inversion we add lines of zeros at adequate places
            A_inv_final = np.zeros((nb_spec, len(balancing_species)))
            for i, spec in enumerate(balancing_species):
                A_inv_final[spec_names.index(spec),:] = A_reduced_inv[i,:]
                
            # We will need the transpose
            self.A_inv_final_t = np.transpose(A_inv_final)
        
            # Indices of constrained species used in keras (remark: correct order is kept)
            #cst_spec_indices = [spec_names.index(spec) for spec in balancing_species]
        
        
        # More natural method : orthogonal projection    /!\ NOT OPERATIONAL /!\
        if self.hard_constraints_model==2:

            # We need to form matrix L
            M = A_atomic
            MMt = np.dot(M, np.transpose(M))
            MMt_inv = np.linalg.inv(MMt)
            MMt_inv_M = np.dot(MMt_inv, M)
            N = np.dot(np.transpose(M),MMt_inv_M)
            self.L = np.eye(N.shape[0]) - N

    def get_scaler_params(self, scaler):
    
        param1 = scaler.mean_
        param2 = scaler.var_
            
        return param1, param2

    def check_inputs(self):

        # Mechanism type
        if self.mechanism_type not in ["detailed", "reduced"]:
            sys.exit('ERROR: mechanism_type should be either "detailed" or "reduced"')

        # Layers type
        if self.layers_type not in ["dense", "resnet"]:
            sys.exit("ERROR: Wrong layer type, it should be 'dense' or 'resnet'")

        # Conservation layer
        if self.hard_constraints_model==2 and (not self.output_omegas):
            sys.exit("hard_constraints_model=2 not written for output_omegas=False")

