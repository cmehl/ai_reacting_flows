import sys

import numpy as np
import cantera as ct

import ai_reacting_flows.tools.utilities as utils

# =============================================================================
# INLET CLASS TO DEFINE STOCHASTIC REACTORS INITIAL STATE
# =============================================================================

class Inlet(object):
    
    def __init__(self, inlet_type, nb_particles):
        
        self.inlet_type = inlet_type
        self.nb_particles = nb_particles
        
        if inlet_type=="blank":
            self.set_state = self.set_state_blank
        elif inlet_type=="cold_premixed":
            self.set_state = self.set_state_cold_premixed
        elif inlet_type=="burnt_premixed":
            self.set_state = self.set_state_burnt_premixed
        else:
            sys.exit("ERROR Inlet type must be one of the following: \n - blank \n - cold_premixed \n - burnt_premixed")
        
        
    def set_state_blank(self, mech, T, p):
        
        # Cantera gas object
        gas = ct.Solution(mech)
        
        # Number of columns in inlet file: nb parts, T, p, species
        nb_cols = gas.n_species + 3
        
        # Creating state with species set to 0
        state = np.zeros(nb_cols)
        state[0] = self.nb_particles
        state[1] = T
        state[2] = p
        
        self.state = state
        
        
    def set_state_cold_premixed(self, fuel, mech, phi, T, p):
        
        # FIRST GET VALUES FOR Y_fuel, Y_N2 and Y_O2
        # Parse fuel species (Remark: might include oxygen)
        x, y, z, _= utils.parse_species(fuel)
            
        # Molar mass of elements (in g/mol)
        W_C = 12.011
        W_H = 1.008
        W_O = 15.999
        W_N = 14.007
    
        # Molar mass of species
        W_fuel, W_O2, W_N2 = x*W_C + y*W_H + z*W_O, 2*W_O, 2*W_N
        
        # Total number of moles
        n_tot = (phi + (x + y/4 - z/2)*4.76)
    
        # Molar fractions
        X_fuel, X_O2, X_N2 = phi/n_tot, (x + y/4 - z/2)/n_tot, 3.76*(x + y/4 - z/2)/n_tot
        
        # Molar mass of the mixture
        W = X_fuel*W_fuel + X_O2*W_O2 + X_N2*W_N2
        
        # desired mass fractions
        Y_fuel, Y_O2, Y_N2 = (W_fuel/W) * X_fuel, (W_O2/W) * X_O2, (W_N2/W) * X_N2
    
    
        # FILL STATE
        # Cantera gas object
        gas = ct.Solution(mech)
        
        # Number of columns in inlet file: nb parts, T, p, species
        nb_cols = gas.n_species + 3
        
        # Creating state with species set to 0
        state = np.zeros(nb_cols)
        state[0] = self.nb_particles
        state[1] = T
        state[2] = p
        state[3 + gas.species_index(fuel)] = Y_fuel
        state[3 + gas.species_index("O2")] = Y_O2
        state[3 + gas.species_index("N2")] = Y_N2
        
        self.state = state
        
        
        
    def set_state_burnt_premixed(self, fuel, mech, phi, T, p):
        
        # COMPUTE BURNT GAS STATE
        # Cantera gas object
        gas = ct.Solution(mech)
        
        # Determining initial composition using phi (fuel + air)
        fuel_ox_ratio = gas.n_atoms(species=fuel, element='C') \
                        + 0.25 * gas.n_atoms(species=fuel, element='H') \
                        - 0.5 * gas.n_atoms(species=fuel, element='O')
        compo = f'{fuel}:{phi:3.2f}, O2:{fuel_ox_ratio:3.2f}, N2:{fuel_ox_ratio * 0.79 / 0.21:3.2f}'
        
        # Equilibrium computation
        gas.TPX = T, p, compo
        gas.equilibrate('HP')
        
        # FILL STATE        
        # Number of columns in inlet file: nb parts, T, p, species
        nb_cols = gas.n_species + 3
        
        # Creating state with species set to 0
        state = np.zeros(nb_cols)
        state[0] = self.nb_particles
        state[1] = gas.T
        state[2] = gas.P
        state[3:] = gas.Y
        
        self.state = state