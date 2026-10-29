#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:22:15 2019

@author: mehlc
"""

import sys
import numpy as np
from scipy.stats import gaussian_kde
import cantera as ct

#==============================================================================
# COMBUSTION RELATED FUNCTIONS
# =============================================================================

# Function to compute equivalence ratio from composition
def Phi_from_Yk_HC(n, m, Y_HC, Y_O2, Y_N2):

    """This function computes the equivalence ratio of an air - hydrocarbon mixture.

    The composition is computed using the mixture description: Phi CnHm + (n+m/4)(O2+3.76 N2)
    - First, X_CnHm is written as a function of Phi
    - Then relationship is inverted
    - Finally the equivalence ratio is computed by writing that X_CnHm=(W/Wk)Y_CnHm are calculated as Yk=(Wk/W)Xk

    Parameters
    ----------
    n : int
        Number pf carbon atoms in the hydrocarbon
    m : int
        Number of hydrogen atoms in the hydrocarbon
    Y_HC: float
        Mass fraction of hydrocarbon
    Y_O2: float
        Mass fraction of oxygen
    Y_N2: float
        Mass fraction of nitrogen

    Returns
    -------
    float
        Equivalence ratio of the mixture

    """

    #==========================================================================
    #     QUANTITIES ASSOCIATED TO SPECIES
    #==========================================================================

    # Molar mass of elements (in g/mol)
    W_C = 12.011
    W_H = 1.008
    W_O = 15.999
    W_N = 14.007

    # Molar mass of species
    W_HC = n*W_C+m*W_H
    W_O2 = 2*W_O
    W_N2 = 2*W_N

    #==========================================================================
    #     MOLE FRACTION OF FUEL
    #==========================================================================

    # Molar mass of the mixture
    W = 1.0 / ( Y_HC/W_HC + Y_O2/W_O2 + Y_N2/W_N2)

    # Hydrocarbon mole fraction
    X_HC = (W/W_HC)*Y_HC

    #==========================================================================
    #     EQUIVALENCE RATIO
    #==========================================================================

    Phi = (X_HC*(n +(m/4))*(1.0+3.76)) / (1 - X_HC)


    return Phi

# Function to compute composition from equivalence ratio
def Yk_from_phi_HC(fuel, phi, gas):
    """Mass fractions [-] of a fuel/air mixture at equivalence ratio phi [-].

    The fuel CxHyOzNw may contain C, H, O and N. Mixture: phi CxHyOzNw + s (O2 + 3.76 N2),
    with stoichiometric coefficient s = x + y/4 - z/2 (products CO2, H2O and N2).
    Atoms and molar masses [kg/kmol] are taken from the Cantera Solution gas.
    """

    # Atoms of fuel species (Remark: might include oxygen and nitrogen)
    x, y, z, w = get_atomic_array(gas)[:, gas.species_index(fuel)]

    # Molar mass of species
    W_fuel = gas.molecular_weights[gas.species_index(fuel)]
    W_O2 = gas.molecular_weights[gas.species_index("O2")]
    W_N2 = gas.molecular_weights[gas.species_index("N2")]
    
    
    # Total number of moles
    n_tot = (phi + (x + y/4 - z/2)*4.76)


    # Fuel molar fraction
    X_fuel = phi/n_tot
    
    # O2 molar fraction
    X_O2 = (x + y/4 - z/2)/n_tot
    
    # N2 molar fraction
    X_N2 = 3.76*(x + y/4 - z/2)/n_tot
    
    # Molar mass of the mixture
    W = X_fuel*W_fuel + X_O2*W_O2 + X_N2*W_N2
    
    # Fuel mass fraction
    Y_fuel = (W_fuel/W) * X_fuel

    # O2 mass fraction
    Y_O2 = (W_O2/W) * X_O2

    # N2 mass fraction
    Y_N2 = (W_N2/W) * X_N2

    compo = {}
    compo[fuel] = Y_fuel
    compo["O2"] = Y_O2
    compo["N2"] = Y_N2

    return compo

# Elements handled by the code (order of rows of atomic arrays)
ELEMENTS = ["C", "H", "O", "N"]

def check_elements(gas):
    """Raise an error if a species of the Cantera Solution gas contains atoms other than C, H, O and N.

    Inert species such as Ar and He are not handled: use a mechanism without them (e.g. mech_nh3_noArHe.yaml).
    """

    other_elements = [el for el in gas.element_names if el not in ELEMENTS]
    species = [spec for k, spec in enumerate(gas.species_names) if any(gas.n_atoms(k, el) > 0 for el in other_elements)]

    if species:
        raise ValueError(f"Species {species} contain atoms other than C, H, O and N, which are not handled: use a mechanism without them")

def get_atomic_array(gas):
    """Number of atoms [-] of C, H, O, N (rows) in each species (columns) of the Cantera Solution gas."""

    check_elements(gas)

    atomic_array = np.zeros((len(ELEMENTS), gas.n_species))
    for j, el in enumerate(ELEMENTS):
        if el in gas.element_names:
            atomic_array[j,:] = [gas.n_atoms(k, el) for k in range(gas.n_species)]

    return atomic_array

def get_atomic_weights():
    """Atomic weights [kg/kmol] of C, H, O, N (from Cantera)."""

    return np.array([ct.Element(el).weight for el in ELEMENTS])

def get_elements_matrix(gas):
    """Matrix A [-] such that A @ Y gives the mass fractions of C, H, O, N: A_jk = n_jk W_j / W_k."""

    return get_atomic_weights()[:, None] * get_atomic_array(gas) / gas.molecular_weights

# Function to get matrix (Wj/Wk)*n_k^j   (Remark: order of atoms is C, H, O, N)
def get_molar_mass_atomic_matrix(gas, with_N2_chemistry):
    """Matrix A [-] such that A @ Y gives the elements mass fractions, and the elements names.

    Rows: elements present in the species of the Cantera Solution gas (order C, H, O, N),
    N removed if with_N2_chemistry is False.
    Columns: species (N2 removed if with_N2_chemistry is False).
    """

    A_element = get_elements_matrix(gas)

    # N2 removed from species without N chemistry
    if not with_N2_chemistry:
        A_element = np.delete(A_element, gas.species_index("N2"), axis=1)

    # Keep only elements present in species (N not considered without N chemistry)
    kept = [j for j in range(len(ELEMENTS)) if A_element[j,:].any()]
    if not with_N2_chemistry and 3 in kept:
        kept.remove(3)

    return A_element[kept,:], [ELEMENTS[j] for j in kept]


def compute_X_element(gas, Yk):
    """Elements mole fractions [-] (C, H, O, N) for mass fractions Yk of the species of the Cantera Solution gas."""

    # Xk
    Wk = gas.molecular_weights
    W = 1.0 / np.sum(Yk/Wk)
    Xk = (W/Wk)*Yk

    atomic_array = get_atomic_array(gas)
    X_el = np.dot(atomic_array, Xk)

    return X_el


def get_nasa7_coefficients(gas):
    """NASA 7-coefficient polynomials of the species of the Cantera Solution gas (2 temperature ranges):
    T_mid [K] (n_species), a_low and a_high (n_species, 7), coefficients below and above T_mid.
    Only species with NASA 7-coefficient polynomials (NasaPoly2) are handled."""

    for spec in gas.species():
        if not isinstance(spec.thermo, ct.NasaPoly2):
            raise ValueError(f"Species {spec.name}: only NASA 7-coefficient polynomials are handled")

    # Cantera order of coefficients: T_mid, 7 coefficients above T_mid, 7 coefficients below T_mid
    coeffs = np.array([spec.thermo.coeffs for spec in gas.species()])

    return coeffs[:, 0], coeffs[:, 8:15], coeffs[:, 1:8]


def species_h_cp(T, W_k, T_mid, a_low, a_high):
    """Mass enthalpies h_k [J/kg] and heat capacities cp_k [J/kg/K] (n, n_species) of the species at
    temperatures T [K] (n), from their NASA 7-coefficient polynomials (see get_nasa7_coefficients), with
    W_k: molar masses [kg/kmol] (n_species). Ideal gas: they do not depend on pressure. Vectorized
    equivalent of the Cantera properties of a SolutionArray, which loops over the states.

        cp_k / r_k = a0 + a1 T + a2 T^2 + a3 T^3 + a4 T^4
        h_k / r_k = a0 T + a1 T^2/2 + a2 T^3/3 + a3 T^4/4 + a4 T^5/5 + a5,     r_k = R / W_k

    (McBride, Gordon & Reno, NASA TM-4513, 1993, https://ntrs.nasa.gov/citations/19940013151)
    """

    T = np.asarray(T, dtype=float)[:, None]
    r_k = ct.gas_constant / W_k

    # Polynomials of one temperature range, for all states and species
    def polynomials(a):
        cp = a[:, 0] + T * (a[:, 1] + T * (a[:, 2] + T * (a[:, 3] + T * a[:, 4])))
        h = T * (a[:, 0] + T * (a[:, 1] / 2.0 + T * (a[:, 2] / 3.0 + T * (a[:, 3] / 4.0 + T * a[:, 4] / 5.0)))) + a[:, 5]
        return cp, h

    cp_low, h_low = polynomials(a_low)
    cp_high, h_high = polynomials(a_high)

    # Range of each species at each temperature (as in Cantera: low range up to T_mid included)
    high = T > T_mid
    h_k = r_k * np.where(high, h_high, h_low)
    cp_k = r_k * np.where(high, cp_high, cp_low)

    return h_k, cp_k

#==============================================================================
# CANONICAL FLAMES COMPUTATION
# =============================================================================

def compute_adiabatic(fuel, mech_file, phi, T0, p, diffusion_model="Mix"):
        
    #-----------------------------------------------
    #           FLAME COMPUTATION
    #-----------------------------------------------
    
    initial_grid = np.linspace(0.0, 0.03, 10)  # m
    tol_ss = [1.0e-4, 1.0e-9]  # [rtol atol] for steady-state problem
    tol_ts = [1.0e-5, 1.0e-5]  # [rtol atol] for time stepping
    loglevel = 1  # amount of diagnostic output (0 to 8)
    
    # IdealGasMix object used to compute mixture properties, set to the state of the
    # upstream fuel-air mixture
    gas = ct.Solution(mech_file)
    
    # First flame equivalence ratio
    fuel_ox_ratio = gas.n_atoms(fuel,'C') + 0.25*gas.n_atoms(fuel,'H') - 0.5*gas.n_atoms(fuel,'O')
    reactants = fuel + ':{:3.2f}, O2:{:3.2f}, N2:{:3.2f}'.format(phi,fuel_ox_ratio,fuel_ox_ratio * 0.79/0.21) 
    
    
    # Flame object
    gas.TPX = T0, p, reactants
    f = ct.FreeFlame(gas, initial_grid)

    f.flame.set_steady_tolerances(default=tol_ss)
    f.flame.set_transient_tolerances(default=tol_ts)
    f.inlet.set_steady_tolerances(default=tol_ss)
    f.inlet.set_transient_tolerances(default=tol_ts)
    f.outlet.set_steady_tolerances(default=tol_ss)
    f.outlet.set_transient_tolerances(default=tol_ts)
        
    # Mixing model and Jacobian
    f.transport_model = diffusion_model
    f.set_max_jac_age(10, 10)
    f.set_time_step(1e-5, [2, 5, 10, 20])
    
    # # Solve with the energy equation enabled
    # f.energy_enabled = True
    # f.set_refine_criteria(ratio=3, slope=0.1, curve=0.5)
    # f.solve(loglevel=loglevel, refine_grid=True)
    # print('Flamespeed = {0:7f} m/s'.format(f.velocity[0]))
    
    # f.set_refine_criteria(ratio=3, slope=0.05, curve=0.2)
    # f.solve(loglevel=loglevel, refine_grid=True)
    # print('Flamespeed = {0:7f} m/s'.format(f.velocity[0]))
    
    # f.set_refine_criteria(ratio=3, slope=0.05, curve=0.1, prune=0.03)
    # #f.transport_model = 'Multi'
    # f.solve(loglevel=loglevel, refine_grid=True)

    f.set_refine_criteria(ratio=2, slope=0.06, curve=0.12, prune=0.04)
        
    f.solve(loglevel=loglevel, auto=True)

    # Getting needed variables
    T = f.T
    Y_dict = dict.fromkeys(gas.species_names)
    for spec in gas.species_names:
        Y_dict[spec] = f.Y[gas.species_index(spec),:]
        
    return T, Y_dict

def compute_0D_reactor(fuel, mech_file, phi, T0, p):
    
    gas = ct.Solution(mech_file)

    # Mixture composition
    fuel_ox_ratio = gas.n_atoms(fuel,'C') + 0.25*gas.n_atoms(fuel,'H') - 0.5*gas.n_atoms(fuel,'O')
    reactants = fuel + ':{:3.2f}, O2:{:3.2f}, N2:{:3.2f}'.format(phi,fuel_ox_ratio,fuel_ox_ratio * 0.79/0.21)
    gas.TPX = T0, p , reactants
    
    # Equilibrate the gas mixture
    gas_equil = ct.Solution(mech_file)
    gas_equil.TPX = T0, p , reactants
    gas_equil.equilibrate('HP')
   
    # Save the equilibrium state 
    state_equil = np.append(gas_equil.X, gas_equil.T)  

    simtime = 0
   
    # we suppose that equilibrium is reached before 1000 time steps
    dt = 5.0e-7
    max_time = 1000*dt 
   
    # tolerance for equilbirium 
    tol_equilibirium = 0.5  # in %
   
    # Advance the simulation until the equilbrium is reached within 0.5% error range
    equil_bool = False
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    simtime = 0.0
    states = ct.SolutionArray(gas, extra=['t'])
    while (equil_bool == False) and (simtime <max_time):
       
        simtime += dt
        sim.advance(simtime)
        states.append(r.thermo.state, t=simtime*1e3)

        # Checking if equilibrium is reached
        state_current = np.append(r.thermo.X, r.T)
        residual = 100.0 * np.linalg.norm(state_equil - state_current, ord=np.inf) / np.linalg.norm(state_equil,
                                                                                                   ord=np.inf)
        if residual < tol_equilibirium:
            equil_bool = True
    
    # Getting needed variables
    T = states.T
    Y_dict = dict.fromkeys(gas.species_names)
    for spec in gas.species_names:
        Y_dict[spec] = states.Y[:,gas.species_index(spec)]
    
    return T, Y_dict



# =============================================================================
# MATHEMATICAL UTILITIES
# =============================================================================

def sample_pairs(n, nsamp):

    max_pairs = n * (n - 1) // 2

    # sample unique "ranks" in the space of all i<j pairs
    ranks = np.random.choice(max_pairs, nsamp, replace=False)

    # unrank: convert a linear index into (i, j) with i < j
    # using the standard combinatorial unranking for k=2
    i = (np.floor((2 * n - 1 - np.sqrt((2 * n - 1) ** 2 - 8 * ranks)) / 2)).astype(np.int64)
    j = ranks - i * (2 * n - i - 1) // 2 + i + 1

    return np.vstack([i, j]).T


# Compute PDF using KDE method
def compute_pdf(x, samples):

    pdf_kde = gaussian_kde(samples)
    pdf = pdf_kde(x)

    return pdf

#==============================================================================
# MISC
#==============================================================================

# Print with flush
def PRINT(*args):
    sys.stdout.write(" ".join(["%s"%arg for arg in args])+"\n")
    sys.stdout.flush()



def smart_binwidth(x, fallback_bins=10):
        """
        Compute a data-driven bin width using the Freedman-Diaconis rule:
            binwidth = 2 * IQR(x) / n^(1/3)
        Falls back to a fixed number of bins if IQR is zero or sample is too small.
        """

        x = np.asarray(x)
        x = x[~np.isnan(x)]
        n = x.size

        if n < 2:
            return 1.0

        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        data_range = x.max() - x.min()

        if iqr <= 0 or data_range <= 0:
            # Degenerate/near-constant data: just split range into fixed bins
            return (data_range / fallback_bins) if data_range > 0 else 1.0

        binwidth = 2 * iqr / (n ** (1 / 3))

        # Guard against pathologically small binwidth relative to range
        # (e.g. huge n with tight IQR) by capping the number of bins
        min_binwidth = data_range / (fallback_bins * 10)
        binwidth = max(binwidth, min_binwidth)

        return binwidth
