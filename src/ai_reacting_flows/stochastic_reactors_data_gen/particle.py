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

    def __init__(self, state_ini, species_names, num_inlet, num_part, data_gen_parameters, activation_time, Yc_u_ini, parent : 'ParticlesCloud'):
        
        # Inlet number from which it is issued
        self.num_inlet = num_inlet
        
        # Particle numerotation
        self.num_part = num_part

        # Particle activation time (before that time it is frozen)
        self.activation_time = activation_time
        
        # Chemical mechanism
        self.mech_file = data_gen_parameters["mech_file"]

        # Fuel (only single fuel for the mixture fraction)
        self.fuel = data_gen_parameters["fuel"][0]

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
        
        # Number of atoms (rows: C, H, O, N) in each species and atomic weights [kg/kmol], from Cantera
        self.species_atoms = utils.get_atomic_array(parent.gas)
        self.W_atoms = utils.get_atomic_weights()
        
        # Compute initial equivalence ratio
        self.compute_equiv_ratio()
        
        # Compute initial mixture fraction
        self.compute_mixture_fraction()
        
        #
        self.Yc_u = Yc_u_ini
        self.recompute_Yc_eq = True  # We initially need to compute Yc_eq

        # Age property of the EMST_POPE mixing model [-] (see emst_pope.py): remaining time of the mixing
        # (> 0) or non-mixing (< 0) state, in units of mixing_time. Set by ParticlesCloud, NaN for other models.
        self.emst_age = np.nan
        self.compute_progress_variable(parent)
        #
        self.compute_heat_release_rate(parent)
        
# =============================================================================
#     PARTICLE CHEMICAL EVOLUTION
# =============================================================================
    
    # Standard (Cantera) function to update chemistry
    def react(self, dt, parent : 'ParticlesCloud'):
        """Advance particle chemistry by dt with a constant-pressure Cantera reactor.
        """
        
        # Unsolved pb: if Yk from NN is inputed here; 
        # it may become negative and mass is lost (output from CVODE is always positive)

        if self.T <= parent.T_threshold:
            return
        
        # Initial value are current's particle state
        parent.gas.HPY = self.hs, self.P, self.Y
        
        # Constant pressure reactor
        r = ct.IdealGasConstPressureReactor(parent.gas)
        
        # Initializing reactor
        sim = ct.ReactorNet([r])
        
        # Advancing to dt
        try:
            sim.advance(dt)
        except ct.CanteraError as e:
            raise RuntimeError(f"Cantera integration failed: {e}") from e
        
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


    def set_state_from_T_Y(self, T, Y, parent : 'ParticlesCloud'):
        """Set the particle state from temperature T [K] and mass fractions Y [-] at the particle
        pressure, e.g. after an ANN chemistry step. Y is clipped to positive values and normalized
        by Cantera."""

        # Cantera sets negative mass fractions to 0 and normalizes them to sum 1
        parent.gas.TPY = T, self.P, Y
        self.T = T
        self.Y = parent.gas.Y
        self.X = parent.gas.X

        # Enthalpy from temperature
        self.compute_hs_from_T(parent)

        state_new = np.empty(self.nb_state_vars)
        state_new[0] = self.hs
        state_new[1] = self.P
        state_new[2:] = self.Y
        self.state = state_new

        # We need to update masses of species and enthalpies (total mass in reactor is preserved)
        self.mass_k = self.mass * self.Y
        self.Hs = self.mass * self.hs

    
# =============================================================================
#     FUNCTIONS TO COMPUTE COMPOSITION-DERIVED QUANTITIES
# =============================================================================

    # Equivalence ratio based on atomic balance
    def compute_equiv_ratio(self):
        """Equivalence ratio based on atomic (C, H, O) balance."""

        # Atoms C, H, O of each species ([:3] because N is not considered here)
        local_c, local_h, local_o = self.species_atoms[:3] @ self.X

        o_stoich = 2.0 * local_c + 0.5 * local_h
        self.equiv_ratio = o_stoich / (local_o + 1.0e-20)


    def _compute_Yc_eq(self, parent : 'ParticlesCloud'):
        """Equilibrium progress-variable species mass fraction at current (T, P, X)."""

        # Equilibrium state in present conditions
        parent.gas.TPX = self.T, self.P, self.X
        parent.gas.equilibrate('HP')
        self.Yc_eq = parent.gas.Y[parent.pv_ind].sum()

    
    # Equivalence ratio based on atomic balance
    def compute_progress_variable(self, parent : 'ParticlesCloud'):
        """Progress variable based on true chemical equilibrium."""

        # We compute Yc_eq only if particle has mixed in current time step (the value stays the same for reacting-only particles)
        if self.recompute_Yc_eq:
            self._compute_Yc_eq(parent)
            self.recompute_Yc_eq = False # now that it is computed we can reset flag to False

        # Initializing
        self.prog_var = 0.0
        
        Yc = self.Y[parent.pv_ind].sum()

        denom = self.Yc_eq - self.Yc_u
        self.prog_var = (Yc - self.Yc_u) / denom if denom > 1e-10 else 0.0
            

    # Heat release rate
    def compute_heat_release_rate(self, parent : 'ParticlesCloud'):
        """Heat release rate = -sum_k( wdot_k * h_k^0 )."""

        parent.gas.TPX = self.T, self.P, self.X

        # Vectorized over all species at once (both arrays already indexed
        # by species order, no need for repeated species_index lookups)
        standard_enthalpies = parent.gas.standard_enthalpies_RT * ct.gas_constant * parent.gas.T
        self.hrr = -np.dot(parent.gas.net_production_rates, standard_enthalpies)


    # Mixture fraction based on atomic balance
    def compute_mixture_fraction(self):
        """Mixture fraction Z [-] based on atomic (C, H) mass balance.

        Z = (Y_C + Y_H) / (Y_C + Y_H)_fuel, with Y_C, Y_H the elements mass fractions: 0 in air,
        1 in pure fuel, for any fuel CxHyOzNw (e.g. NH3: (Y_C + Y_H)_fuel = 3 W_H / W_NH3).
        Assumes the oxidizer contains no C and H (air).
        """

         # Precompute per-species C/H mass contribution coefficients once;
        # only self.Y changes call-to-call, not composition or molar masses.
        if not hasattr(self, '_mixfrac_coeff'):
            n_C, n_H = self.species_atoms[0], self.species_atoms[1]
            coeff = (n_C * self.W_atoms[0] + n_H * self.W_atoms[1]) / self.spec_mol_weights

            # Normalization by the C and H mass fraction of the pure fuel
            self._mixfrac_coeff = coeff / coeff[self.species_names.index(self.fuel)]

        self.mix_frac = np.dot(self._mixfrac_coeff, self.Y)

        
# =============================================================================
#     FUNCTIONS TO COMPUTE THERMODYNAMICS QUANTITIES
# =============================================================================
    
    # Sensible enthalpy from temperature
    def compute_hs_from_T(self, parent : 'ParticlesCloud'):
        """Sensible enthalpy from temperature, pressure, and composition."""

        # Initial value are current's particle state
        parent.gas.TPY = self.T, self.P, self.Y
        
        # Enthalpy
        self.hs = parent.gas.HP[0]
        

    # Update states from H/Y after diffusion
    def update_ThermoStates(self, parent : 'ParticlesCloud'):
        """Update T, X from current H, P, Y state (e.g. after a diffusion step)."""

        # Initial value are current's particle state
        parent.gas.HPY = self.hs, self.P, self.Y
        
        # Add required states if needed
        self.T = parent.gas.T
        self.X = parent.gas.X


    def compute_lewis_numbers(self, parent : 'ParticlesCloud'):
        """Species Lewis numbers: Le_k = lambda / (rho * cp * D_k)."""

        parent.gas.TPY = self.T, self.P, self.Y

        # Lewis numbers
        cond = parent.gas.thermal_conductivity
        cp = parent.gas.cp_mass
        Dk =  parent.gas.mix_diff_coeffs_mass
        rho = parent.gas.density
        # mu = parent.parent.gas.viscosity
        #
        self.Le_k = cond / (rho*cp*Dk)
