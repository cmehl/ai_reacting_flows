import numpy as np
import pandas as pd
import cantera as ct

from ai_reacting_flows.stochastic_reactors_data_gen.pre_processing import Inlet

mech_file = "../../data/chemical_mechanisms/mech_H2.yaml"

fuel = "H2"

inlet_1 = Inlet("cold_premixed", nb_particles=500)
inlet_2 = Inlet("cold_premixed", nb_particles=100)
    
inlet_1.set_state(fuel=fuel, mech=mech_file, phi=0.4, T=300.0, p=101325.0)
inlet_2.set_state(fuel=fuel, mech=mech_file, phi=0.4, T=1200.0, p=101325.0)

list_inlets = [inlet_1, inlet_2]
nb_inlets = len(list_inlets)

gas = ct.Solution(mech_file)
thermo_chem_vars = gas.species_names
thermo_chem_vars.insert(0, "P")
thermo_chem_vars.insert(0, "T")
thermo_chem_vars.insert(0, "Np")
nb_vars = len(thermo_chem_vars)
    
inlet_names = [f"inlet_{i}" for i in range(1,nb_inlets+1)]
    
data = np.zeros((nb_inlets, nb_vars))
i = 0
for inlet in list_inlets:
    data[i, :] = inlet.state
    i += 1
    
df = pd.DataFrame(data, columns = thermo_chem_vars, index = inlet_names)

df.to_excel("inlets_file.xlsx")