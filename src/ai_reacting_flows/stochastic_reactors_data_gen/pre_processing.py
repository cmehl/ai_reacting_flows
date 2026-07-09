import sys

import numpy as np
import cantera as ct

import ai_reacting_flows.tools.utilities as utils

# =============================================================================
# INLET CLASS TO DEFINE STOCHASTIC REACTORS INITIAL STATE
# =============================================================================

class Inlet(object):
    
    def __init__(self, inlet_type, nb_particles, activation_time, pv_ind):
        
        self.inlet_type = inlet_type
        self.nb_particles = nb_particles
        self.activation_time = activation_time
        self.pv_ind = pv_ind

        self.Yc_u = 0.0

        state_setters = {
            "blank": self.set_state_blank,
            "cold_premixed": self.set_state_cold_premixed,
            "burnt_premixed": self.set_state_burnt_premixed,
            "air": self.set_state_air,
            "fuel": self.set_state_fuel,
        }

        try:
            self.set_state = state_setters[inlet_type]
        except KeyError:
            raise ValueError(
                f"Invalid inlet_type '{inlet_type}'. Must be one of: "
                f"{', '.join(state_setters)}"
            )
        
        
    def set_state_blank(self, mech, T, P):
        """
        Initialize self.state as a particle state with all species mass
        fractions set to zero (e.g. placeholder / inert particle).

        Parameters
        ----------
        mech : str   Path to the Cantera mechanism file.
        T    : float Temperature [K].
        P    : float Pressure [Pa].
        """
        gas = ct.Solution(mech)

        Y = np.zeros(gas.n_species)

        # State layout: [nb_particles, T, P, Y_0, ..., Y_nsp-1]
        self.state = np.concatenate(([self.nb_particles, T, P], Y))
        self.Yc_u = 0.0
        
        
    def set_state_cold_premixed(self, fuel, mech, phi, T, P):
        """
        Build a cold (unburnt) premixed particle state at equivalence ratio `phi`.

        Sets self.state = [nb_particles, T, P, Y_0, ..., Y_nsp-1]
        and self.Yc_u    = sum of progress-variable species mass fractions
                            in the unburnt mixture.

        Parameters
        ----------
        fuel : str   Fuel species name (must exist in `mech`).
        mech : str   Cantera mechanism file.
        phi  : float Equivalence ratio.
        T    : float Temperature [K].
        P    : float Pressure [Pa].
        """

        gas = ct.Solution(mech)
        
        if fuel not in gas.species_names:
            raise ValueError(f"Fuel '{fuel}' not found in mechanism '{mech}'")
        
        # Let Cantera handle the stoichiometry (robust to fuels containing
        # O or N atoms, multi-species fuel blends, etc.)
        gas.TP = T, P
        gas.set_equivalence_ratio(phi, fuel, {"O2": 1.0, "N2": 3.76})
        Y = gas.Y

        # State layout: [nb_particles, T, P, Y_0, ..., Y_nsp-1]
        self.state = np.concatenate(([self.nb_particles, T, P], Y))

        self.Yc_u = Y[self.pv_ind].sum()
        
        
        
    def set_state_burnt_premixed(self, fuel, mech, phi, T, P):
        """
        Build an equilibrium (burnt) premixed particle state at equivalence ratio `phi`.

        Sets self.state = [nb_particles, T_burnt, P, Y_0, ..., Y_nsp-1]
        and self.Yc_u    = sum of progress-variable species mass fractions
                            in the *unburnt* mixture.

        Parameters
        ----------
        fuel : str   Fuel species name (must exist in `mech`).
        mech : str   Cantera mechanism file.
        phi  : float Equivalence ratio.
        T    : float Unburnt temperature [K].
        P    : float Pressure [Pa].
        """

        # COMPUTE BURNT GAS STATE
        # Cantera gas object
        gas = ct.Solution(mech)

        if fuel not in gas.species_names:
            raise ValueError(f"Fuel '{fuel}' not found in mechanism '{mech}'")
        
        # Set unburnt composition exactly (no string-formatting precision loss,
        # robust to fuels containing O or N atoms)
        gas.TP = T, P
        gas.set_equivalence_ratio(phi, fuel, {"O2": 1.0, "N2": 3.76})

        self.Yc_u = gas.Y[self.pv_ind].sum()

        gas.equilibrate("HP")
        
        self.state = np.concatenate(([self.nb_particles, gas.T, gas.P], gas.Y))


    def set_state_air(self, mech, T, P):
        """
        Build a pure-air particle state.

        Sets self.state = [nb_particles, T, P, Y_0, ..., Y_nsp-1]
        and self.Yc_u    = sum of progress-variable species mass fractions
                            in air (should be 0 unless O2/N2 are progress
                            variable species).

        Parameters
        ----------
        mech : str   Cantera mechanism file.
        T    : float Temperature [K].
        P    : float Pressure [Pa].
        """
    
        # FILL STATE
        # Cantera gas object
        gas = ct.Solution(mech)

        # Mass fractions of standard air (23.3% O2, 76.7% N2 by mass)
        Y = np.zeros(gas.n_species)
        Y[gas.species_index("O2")] = 0.233
        Y[gas.species_index("N2")] = 0.767
        
        self.Yc_u = Y[self.pv_ind].sum()
        
        # State layout: [nb_particles, T, P, Y_0, ..., Y_nsp-1]
        self.state = np.concatenate(([self.nb_particles, T, P], Y))


    def set_state_fuel(self, mech, fuel, T, P):
        """
        Initialize self.state as a pure-fuel particle state vector.

        Parameters
        ----------
        mech : str
            Path to the Cantera mechanism file.
        fuel : str
            Species name to set to a mass fraction of 1.0.
        T : float
            Temperature [K].
        P : float
            Pressure [Pa].

        Raises
        ------
        ValueError
            If `fuel` is not a species in the mechanism.
        """

        # Cantera gas object
        gas = ct.Solution(mech)

        if fuel not in gas.species_names:
            raise ValueError(
                f"Species '{fuel}' not found in mechanism '{mech}'. "
                f"Available species: {gas.species_names}"
            )
        
        # Pure fuel: mass fraction 1.0 for `fuel`, 0 elsewhere
        Y = np.zeros(gas.n_species)
        Y[gas.species_index(fuel)] = 1.0

        self.Yc_u = Y[self.pv_ind].sum()

        # State layout: [nb_particles, T, P, Y_0, ..., Y_nsp-1]
        self.state = np.concatenate(([self.nb_particles, T, P], Y))