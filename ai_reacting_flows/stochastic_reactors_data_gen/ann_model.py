#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:33:52 2020

@author: mehlc

"""

import sys,os

import shelve

import joblib

from tensorflow.keras.models import model_from_json


from ai_reacting_flows.ann_model_generation.tensorflow_custom import AtomicConservation
from ai_reacting_flows.ann_model_generation.tensorflow_custom import AtomicConservation_RR
from ai_reacting_flows.ann_model_generation.tensorflow_custom import AtomicConservation_RR_lsq


class ModelANN(object):
    
# =============================================================================
#     INITIALIZER
# =============================================================================
    
    def __init__(self, model_name):
        
        # EXTERNAL PARAMETERS
        self.model_name = model_name
        self.model_folder = os.environ.get('AI_KINETIC_PATH') + "/data/ML_models/stochastic_MLP/"
        
        # PARAMETERS READING
        # Load shelve file
        shelfFile = shelve.open(self.model_folder + model_name + "/model_params")
        
        # threshold for negligible concentrations
        self.threshold = shelfFile["threshold"]
        
        # Log transform input data or not
        self.log_transform_X = shelfFile["log_transform_X"]
        
        # Log transform output data or not
        self.log_transform_Y = shelfFile["log_transform_Y"]
        
        # Temperature prediction: "network" or "equation"
        self.T_prediction = shelfFile["T_prediction"]
        
        # Temperature prediction: "network" or "equation"
        self.output_omegas = shelfFile["output_omegas"]
        
        # Scaler used for Y (we need that because Y might be unscaled)
        self.scaler_Y = shelfFile["scaler_Y"]
        
        # Enforcing mass conservation in NN
        self.hard_constraints_model = shelfFile["hard_constraints_model"]
        
        # Closing shelve
        shelfFile.close()
        

# =============================================================================
# ANN MODEL LOADING FUNTIONS    
# =============================================================================
        

    def load_scalers(self):
        self.Xscaler = joblib.load(self.model_folder + self.model_name + "/Xscaler.pkl")
        self.Yscaler = joblib.load(self.model_folder + self.model_name + "/Yscaler.pkl") 
         
            
    def load_ann_model(self):
        
        # Model reconstruction from JSON file
        with open(self.model_folder + self.model_name + '/model_architecture.json', 'r') as f:
            if self.hard_constraints_model==1:
                if self.output_omegas==True:
                    self.model = model_from_json(f.read(), custom_objects={'AtomicConservation_RR': AtomicConservation_RR})
                else:
                    self.model = model_from_json(f.read(), custom_objects={'AtomicConservation': AtomicConservation})
            elif self.hard_constraints_model==2:
                if self.output_omegas==True:
                    self.model = model_from_json(f.read(), custom_objects={'AtomicConservation_RR_lsq': AtomicConservation_RR_lsq})
                else:
                    sys.exit('Not implemented yet')
            else:
                self.model = model_from_json(f.read())
        
        # Load weights into the new model
        self.model.load_weights(self.model_folder + self.model_name + '/model_weights.h5')



# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================
        
    def print_simu_parameters(self):
        print("\nParameters of model:")
        print(f">> Threshold for negligible concentrations: {self.threshold}")
        print(f">> Logarithmic transformation of input: {self.log_transform_X}")
        print(f">> Logarithmic transformation of output: {self.log_transform_Y}")
        print(f">> Prediction of temperature: {self.T_prediction}")
        print(f">> Using reaction rates as outputs of neural network: {self.output_omegas}")
        