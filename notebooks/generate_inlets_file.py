# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: 'Python 3.9.5 (''venv'': venv)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Inlets files generation
#
# In this sript we write the commands used to generate the inlets.xlsx files, which specifies the initial reactors of the stochastics reactors generation.
#
# Each inlet is a dictionary with the necessary data. Available inlet types in the current code:
# + **Blank**: Sets the species mass fractions to *0*, to be specified manualy in the excel file by the user afterwards.
# + **Cold_premixed**: Sets a mixture fuel + air at a given equivalence ratio, pressure and temperature. 
# + **Hot_premixed**: Sets a given mixture air + fuel at a given equivalence ratio, pressure and fresh gas temperature to the corresponding burnt gas state.
#
# Note that new inlet types may be defined by the user by coding them in the *stochastic_reactors_data_gen/pre_processing.py* python script.

# %%
import numpy as np
import pandas as pd
import cantera as ct

from ai_reacting_flows.stochastic_reactors_data_gen.pre_processing import Inlet

# %% [markdown]
# The CANTERA mechanism path and the fuel name must first be specified:

# %%
# Chemical mechanism
mech_file = "../data/chemical_mechanisms/mech_H2.yaml"

# Fuel
fuel = "H2"

# %% [markdown]
# Inlets are then defined, each of them being an object of the *Inlet* class. The *nb_particles* argument corresponds to the number of reactors in the inlet. Here, we define an inlet with cold gases as well as an inlet with burnt gases. Their states are then computed by specifying an equivalence ratio *phi*, a fresh gas temperature *T* and a pressure *p*.

# %%
inlet_1 = Inlet("cold_premixed", nb_particles=450)
inlet_2 = Inlet("burnt_premixed", nb_particles=50)
    
inlet_1.set_state(fuel=fuel, mech=mech_file, phi=0.4, T=300.0, p=101325.0)
inlet_2.set_state(fuel=fuel, mech=mech_file, phi=0.4, T=300.0, p=101325.0)

# %% [markdown]
# We finally generate the *inlets_file.xlsx* file by defining the columns name and filling the data from the inlets defined above.

# %%
list_inlets = [inlet_1, inlet_2]
nb_inlets = len(list_inlets)

# Writing inlets.xlsx file
# Creating CANTERA gas object from mechanism to extract species names
gas = ct.Solution(mech_file)
thermo_chem_vars = gas.species_names
    
# Adding pressure
thermo_chem_vars.insert(0, "P")
    
# Adding temperature
thermo_chem_vars.insert(0, "T")
    
# Adding number of particles in the inlet
thermo_chem_vars.insert(0, "Np")
nb_vars = len(thermo_chem_vars)
    
# Inlet names
inlet_names = [f"inlet_{i}" for i in range(1,nb_inlets+1)]
    
# Filling data
data = np.zeros((nb_inlets, nb_vars))
i = 0
for inlet in list_inlets:
    data[i, :] = inlet.state
    i += 1
    
# Creating pandas dataframe
df = pd.DataFrame(data, columns = thermo_chem_vars, index = inlet_names)
    
# Exporting
df.to_excel("inlets_file_test.xlsx")

# %% [markdown]
#
