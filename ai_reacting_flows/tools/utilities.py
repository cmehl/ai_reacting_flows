#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:22:15 2019

@author: mehlc
"""

import sys

import numpy as np
import pandas as pd
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
def Phi_from_Yk_HC(fuel, phi):
        
    # Parse fuel species (Remark: might include oxygen)
    x, y, z, _= parse_species(fuel)
        
    # Molar mass of elements (in g/mol)
    W_C = 12.0107
    W_H = 1.0079
    W_O = 15.9994
    W_N = 14.0067

    # Molar mass of species
    W_fuel = x*W_C + y*W_H + z*W_O
    W_O2 = 2*W_O
    W_N2 = 2*W_N
    
    
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



#TODO: update this function which is false for two-digits atom numbers (cf cvg_generator)
# Function to parse a species name in order to get its atomic composition
def parse_species(species):
    
    i_char = 0
    n_C = 0
    n_H = 0
    n_O = 0
    n_N = 0
    for i_char in range(len(species)):
        
        if species[i_char] in ["C","c"]:
            try:
                n_C += float(species[i_char+1])
            except ValueError:
                n_C += 1
            except IndexError:
                n_C += 1
                
        elif species[i_char] in ["H","h"]:
            try:
                n_H += float(species[i_char+1])
            except ValueError:
                n_H += 1
            except IndexError:
                n_H += 1
                    
        elif species[i_char] in ["O","o"]:
            try:
                n_O += float(species[i_char+1])
            except ValueError:
                n_O += 1
            except IndexError:
                n_O += 1
            
        elif species[i_char] in ["N","n"]:
            try:
                n_N += float(species[i_char+1])
            except ValueError:
                n_N += 1
            except IndexError:
                n_N += 1


    return n_C, n_H, n_O, n_N




def get_molecular_weights(spec_names):
    
    # Atomic masses (order same as atomic_array: C, H, O, N )
    mass_per_atom = np.array([12.011, 1.008, 15.999, 14.007])
    mass_per_atom = mass_per_atom.reshape((4,1))
    
    # Compute atomic array using the above function
    atomic_array = parse_species_names(spec_names)

    # Multiply array composition by masses
    mass_array = np.multiply(mass_per_atom, atomic_array)
    
    # Molecular weights obtained by squashing the array
    mol_weights = np.sum(mass_array, axis=0)
    
    return mol_weights



def parse_species_names(species_list):
    
    nb_species = len(species_list)
    atomic_array = np.empty((4,nb_species))   # 4 comes from the fact that we are dealing with 4 elements: C, H, O and N
    i_spec = 0
    for i_spec in range(nb_species):
        species = species_list[i_spec]
        i_char = 0
        n_C = 0
        n_H = 0
        n_O = 0
        n_N = 0
        for i_char in range(len(species)):
            
            if species[i_char] in ["C","c"]:
                try:
                    n_C += float(species[i_char+1])
                except ValueError:
                    n_C += 1
                except IndexError:
                    n_C += 1
                    
            elif species[i_char] in ["H","h"]:
                try:
                    n_H += float(species[i_char+1])
                except ValueError:
                    n_H += 1
                except IndexError:
                    n_H += 1
                        
            elif species[i_char] in ["O","o"]:
                try:
                    n_O += float(species[i_char+1])
                except ValueError:
                    n_O += 1
                except IndexError:
                    n_O += 1
                
            elif species[i_char] in ["N","n"]:
                try:
                    n_N += float(species[i_char+1])
                except ValueError:
                    n_N += 1
                except IndexError:
                    n_N += 1
      
        atomic_array[:,i_spec] = np.array([n_C,n_H,n_O,n_N])
        i_spec+=1
        
    return atomic_array



#==============================================================================
# CANONICAL FLAMES COMPUTATION
# =============================================================================


def compute_adiabatic(fuel, mech_file, phi, T0, p, diffusion_model="Mix"):
        
    #-----------------------------------------------
    #           FLAME COMPUTATION
    #-----------------------------------------------
    
    initial_grid = np.linspace(0.0, 0.01, 10)  # m
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
    
    # Mixing model and Jacobian
    f.transport_model = diffusion_model
    f.set_max_jac_age(10, 10)
    f.set_time_step(1e-5, [2, 5, 10, 20])
    
    # Solve with the energy equation enabled
    f.energy_enabled = True
    f.set_refine_criteria(ratio=3, slope=0.1, curve=0.5)
    f.solve(loglevel=loglevel, refine_grid=True)
    print('Flamespeed = {0:7f} m/s'.format(f.u[0]))
    
    f.set_refine_criteria(ratio=3, slope=0.05, curve=0.2)
    f.solve(loglevel=loglevel, refine_grid=True)
    print('Flamespeed = {0:7f} m/s'.format(f.u[0]))
    
    f.set_refine_criteria(ratio=3, slope=0.05, curve=0.1, prune=0.03)
    #f.transport_model = 'Multi'
    f.solve(loglevel=loglevel, refine_grid=True)


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

# TODO: check if the sample is really as it should be
def sample_comb2(dims, nsamp):
    idx = np.random.choice(np.prod(dims), nsamp, replace=False)
    return np.vstack(np.unravel_index(idx, dims)).T


#==============================================================================
# MISC
#==============================================================================

# Print with flush
def PRINT(*args):
    sys.stdout.write(" ".join(["%s"%arg for arg in args])+"\n")
    sys.stdout.flush()