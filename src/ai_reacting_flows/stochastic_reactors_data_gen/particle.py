#
# Allow intellisense in VSCode without having cyclic import (TYPE_CHECKING is False at runtime)
#
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ai_reacting_flows.stochastic_reactors_data_gen.particles_cloud import ParticlesCloud

import numpy as np
import cantera as ct
import ai_reacting_flows.tools.utilities as utils
import warnings

#
# Removes potential warning from sklearn (like feature names related messages)
#
def warn(*args, **kwargs):
    pass

warnings.warn = warn

class Particle(object):
# =============================================================================
#     INITIALIZATION
# =============================================================================    

    def __init__(self, state_ini, species_names, num_inlet, num_part, data_gen_parameters, parent : 'ParticlesCloud'):
        
        # Inlet number from which it is issued
        self.num_inlet = num_inlet
        
        # Particle numerotation
        self.num_part = num_part
        
        # Chemical mechanism
        self.mech_file = data_gen_parameters["mech_file"]

        # species names
        self.species_names = species_names
        
        # Number of species
        self.nb_species = len(species_names)
        
        # Number of state variables
        self.nb_state_vars = len(state_ini)
        
        # Get temperature and mass fractions from state_ini
        self.T = state_ini[0]
        self.P = state_ini[1]
        self.Y = state_ini[2:]
        
        # Get initial conditions
        parent.gas.TPY = self.T, self.P, self.Y
        self.X = parent.gas.X # Species mole fractions
        self.hs = parent.gas.HP[0] # Sensible enthalpy

        # We associate a mass to the particle, and initialize it to 1, by convention
        self.mass = 1.0
        # We can then deduce mass for each chemical species in the particle
        self.mass_k = self.mass * self.Y
        # We can then also define a total sensible enthalpy
        self.Hs = self.mass * self.hs
        
        # Particle state: sensible enthalpy + pressure + mass fractions
        self.state = np.empty(self.nb_state_vars)
        self.state[0] = self.hs
        self.state[1] = self.P
        self.state[2:] = self.Y
        
        # Species molecular weights
        self.spec_mol_weights = parent.gas.molecular_weights
        
        self.species_atoms = utils.parse_species_names(self.species_names)

        self.W_atoms = np.array([parent.gas.atomic_weight("C"), parent.gas.atomic_weight("H"), parent.gas.atomic_weight("O"), parent.gas.atomic_weight("N")]) # Order: C, H, O, N
        
        
        # Compute initial equivalence ratio
        self.compute_equiv_ratio()
        
        # Compute initial mixture fraction
        self.compute_mixture_fraction()
        
        # Compute initial progress variable
        self.calc_progvar = data_gen_parameters["calc_progvar"]
        if self.calc_progvar==True:
            self.pv_species = data_gen_parameters["pv_species"]
            self.npvspec = len(self.pv_species)
            #
            #: int: list of progress variable species indices
            self.pv_ind = []
            for i in range(self.npvspec):
                self.pv_ind.append(parent.gas.species_index(self.pv_species[i]))
        #
        self.compute_progress_variable(parent)
        #
        self.compute_heat_release_rate(parent)
        
# =============================================================================
#     PARTICLE CHEMICAL EVOLUTION
# =============================================================================
    
    # Standard (Cantera) function to update chemistry
    def react(self, dt, parent : 'ParticlesCloud'):
        
        # Unsolved pb: if Yk from NN is inputed here; 
        # it may become negative and mass is lost (output from CVODE is always positive)
        
        if self.T>parent.T_threshold:
            
            # Initial value are current's particle state
            parent.gas.HPY = self.hs, self.P, self.Y
            
            # Constant pressure reactor
            r = ct.IdealGasConstPressureReactor(parent.gas)
            
            # Initializing reactor
            sim = ct.ReactorNet([r])
            
            # Advancing to dt
            sim.advance(dt)
            
            # Updated state
            state_new = np.empty(self.nb_state_vars)
            state_new[0] = parent.gas.HP[0]
            state_new[1] = parent.gas.P
            state_new[2:] = parent.gas.Y
            self.state = state_new
            
            # Variables from state
            self.hs = self.state[0]
            self.P = self.state[1]
            self.Y = self.state[2:]
            self.X = parent.gas.X
            self.T = parent.gas.T

            # We need to update masses of species and enthalpies (total mass in reactor is preserved)
            self.mass_k = self.mass * self.Y
            self.Hs = self.mass * self.hs

    # def react_NN_wrapper(self, ML_models, prog_var_thresholds, dt, T_threshold):
                    
    #     # Single model
    #     if len(ML_models)==1:
    #         self.react_NN(ML_models[0], dt, T_threshold)
    #     # Three models
    #     elif len(ML_models)==3:
    #         # if self.prog_var<=0.01 or  self.prog_var>=0.99:
    #         if self.prog_var<prog_var_thresholds[0]:
    #             self.react_NN(ML_models[0], dt, T_threshold)
    #         elif self.prog_var>prog_var_thresholds[0] and self.prog_var<prog_var_thresholds[1]:
    #             self.react_NN(ML_models[1], dt, T_threshold)
    #         else:  # self.prog_var>prog_var_thresholds[1]
    #             self.react_NN(ML_models[2], dt, T_threshold)


    # TO REWRITE COMPLETELY FOR PYTORCH  
    # NN-based function to update chemistry
    # def react_NN(self, parent):
        
    #     if self.T>parent.T_threshold:
            
    #         # Initializing ANN model
    #         ann_model = ModelANN(parent.ML_model)
            
    #         # Loading model
    #         ann_model.load_ann_model()
    
    #         # Load scalers
    #         ann_model.load_scalers()
            
            
    #         # Storing initial values
    #         Y_old = self.Y
                    
    #         # Gas object modification
    #         parent.gas.TPY= self.T, self.P, self.Y
    #         #        
    #         # State vector for NN (temperature + mass fractions => different than in self.state)
    #         state_NN = np.append(self.T,self.Y)
    #         # NN update for Yk's
    #         # if ann_model.log_transform_X:
    #         #     state_NN[state_NN<ann_model.threshold] = ann_model.threshold
    #         log_state = np.zeros(self.nb_species+1)
    #         log_state[0] = state_NN[0]
    #         if ann_model.log_transform_X:
    #             log_state[1:] = np.log(1.0 + state_NN[1:])
    #         else:
    #             log_state[1:] = state_NN[1:]
            
    #         # Removing N2 if omegas outputed
    #         # if ann_model.output_omegas:
    #         #     log_state = np.delete(log_state, 1 + self.species_names.index("N2"))    
            
    #         # input of NN
    #         log_state = log_state.reshape(1, -1)
    #         NN_input = ann_model.Xscaler.transform(log_state)
                    
    #         # New state predicted by ANN
    #         state_new = ann_model.model.predict(NN_input, batch_size=1)
    
    #         # Getting Y and scaling
    #         if ann_model.scaler_Y != "None":
    #             Y_new = ann_model.Yscaler.inverse_transform(state_new)
    #         else:
    #             Y_new = state_new
    #         # Log transform of species
    #         if ann_model.log_transform_Y:
    #             Y_new = np.exp(Y_new) - 1.0
                    
    #         # Adding N2 again
    #         # if ann_model.output_omegas:
    #         #     Y_new = np.insert(Y_new, self.species_names.index("N2"), 0.0)
                    
    #         # If reaction rate outputs from network
    #         if ann_model.output_omegas:
    #             Y_new += Y_old
                    
    #         # Deducing T from energy conservation
    #         # Explicit update
    #         T_new = state_NN[0] - (1/parent.gas.cp)*np.sum(parent.gas.partial_molar_enthalpies/self.spec_mol_weights*(Y_new-Y_old))
    
    
            
    #         # Remark: we assume self.P stays identical
    #         self.T = T_new
    #         self.Y = Y_new.reshape(self.nb_species)
    #         self.compute_hs_from_T()
            
            
    #         # Updated state
    #         state_new = np.empty(self.nb_state_vars)
    #         state_new[0] = self.hs
    #         state_new[1] = self.P
    #         state_new[2:] = self.Y
    #         self.state = state_new
            
    #         # Variables from state
    #         self.X = self.compute_mol_frac()
            
    #         # Clear memory to avoid overflow
    #         # tf.keras.backend.clear_session()

# =============================================================================
#     FUNCTIONS TO COMPUTE COMPOSITION-DERIVED QUANTITIES
# =============================================================================

    # Equivalence ratio based on atomic balance
    def compute_equiv_ratio(self):
        
        local_o=0.0
        local_c=0.0
        local_h=0.0
        for spec in self.species_names:
            idx = self.species_names.index(spec)
            n_C, n_H, n_O, _ = utils.parse_species(spec)
            
            local_c += n_C*self.X[idx]
            local_h += n_H*self.X[idx]
            local_o += n_O*self.X[idx]
            
        o_stoich = 2.0*local_c+0.5*local_h
        
        self.equiv_ratio = o_stoich/(local_o+1.0e-20)
    
    # Equivalence ratio based on atomic balance
    def compute_progress_variable(self, parent : 'ParticlesCloud'):

        # Initializing
        self.prog_var = 0.0
        
        # Progress variable computed using true equilibrium
        if self.calc_progvar:
            
            # Equilibrium state in present conditions
            parent.gas.TPX = self.T, self.P, self.X
            parent.gas.equilibrate('HP')

            Yc_eq = 0.0
            Yc = 0.0
            for i in self.pv_ind:
                Yc_eq += parent.gas.Y[i]
                Yc += self.Y[i]
            
            self.prog_var = Yc/Yc_eq

    # Heat release rate
    def compute_heat_release_rate(self, parent : 'ParticlesCloud'):

        parent.gas.TPX = self.T, self.P, self.X

        self.hrr = 0.0       
        for spec in parent.gas.species_names:
            standard_enthalpy_spec = parent.gas.standard_enthalpies_RT[parent.gas.species_index(spec)] * ct.gas_constant * parent.gas.T
            self.hrr += -parent.gas.net_production_rates[parent.gas.species_index(spec)] * standard_enthalpy_spec

    # Mixture fraction based on atomic balance
    def compute_mixture_fraction(self):
        
        mix_frac_local = 0.0
        for spec in self.species_names:
            idx = self.species_names.index(spec)
            n_C, n_H, _, _ = utils.parse_species(spec)
            mix_frac_local += n_C * 12.011 * self.Y[idx] / self.spec_mol_weights[idx]
            mix_frac_local += n_H * 1.008  * self.Y[idx] / self.spec_mol_weights[idx]
        
        self.mix_frac = mix_frac_local
        
        
# =============================================================================
#     FUNCTIONS TO COMPUTE THERMODYNAMICS QUANTITIES
# =============================================================================
    
    # Sensible enthalpy from temperature
    def compute_hs_from_T(self, parent : 'ParticlesCloud'):

        # Initial value are current's particle state
        parent.gas.TPY = self.T, self.P, self.Y
        
        # Enthalpy
        self.hs = parent.gas.HP[0]
        
    # Update states from H/Y after diffusion
    def update_ThermoStates(self, parent : 'ParticlesCloud'):
        # Initial value are current's particle state
        parent.gas.HPY = self.hs, self.P, self.Y
        
        # Add required states if needed
        self.T = parent.gas.T
        self.X = parent.gas.X

    def compute_lewis_numbers(self, parent : 'ParticlesCloud'):

        parent.gas.TPY = self.T, self.P, self.Y

        # Lewis numbers
        cond = parent.gas.thermal_conductivity
        cp = parent.gas.cp_mass
        Dk =  parent.gas.mix_diff_coeffs_mass
        rho = parent.gas.density
        # mu = parent.parent.gas.viscosity
        #
        self.Le_k = cond / (rho*cp*Dk)

