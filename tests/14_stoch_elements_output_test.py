"""
Test 14: elements mass fractions in the stochastic reactors output.

Runs a very small H2 stochastic reactors simulation (CURL) mixing an air inlet and a fuel inlet,
so that the elements mass fractions of particles change with time.
Check: Y_C, Y_H, Y_O, Y_N in all_states are the elements mass fractions of the current
composition (compared with Cantera), at every iteration.

Independent of the other tests.
"""
import os
import h5py
import numpy as np
import cantera as ct
import oyaml as yaml
from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database

import DummyComm as dc

# Mechanism file
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

inlet_1 = {"type": "air", "nb_particles": 10, "T": 300.0, "P": 101325.0}
inlet_2 = {"type": "fuel", "nb_particles": 10, "fuel": "H2", "T": 300.0, "P": 101325.0}

data_gen_parameters = {}
data_gen_parameters["mech_file"] = mech_file
data_gen_parameters["inlets"] = {"inlet_1": inlet_1, "inlet_2": inlet_2}
data_gen_parameters["results_folder_suffix"] = "ELEMENTS_TEST"
data_gen_parameters["dtb_file"] = "solutions.h5"
data_gen_parameters["build_ml_dtb"] = False
data_gen_parameters["time_step"] = 1.0e-5
data_gen_parameters["time_max"] = 3.0e-5
data_gen_parameters["automatic_termination"] = False
data_gen_parameters["mixing_model"] = "CURL"
data_gen_parameters["mixing_time"] = 1.0e-5
data_gen_parameters["read_mixing"] = False
data_gen_parameters["T_threshold"] = 200.0
data_gen_parameters["pv_species"] = ["H2O"]
data_gen_parameters["fuel"] = ["H2"]
data_gen_parameters["calc_mean_traj"] = False
data_gen_parameters["ML_inference_flag"] = False


def test_elements_mass_fractions(tmp_path):
    os.chdir(tmp_path)
    with open("dtb_params.yaml", "w") as file:
        yaml.dump(data_gen_parameters, file, default_flow_style=False)

    generate_stochastic_database(dc.DummyComm())

    gas = ct.Solution(mech_file)
    Y_H_all = []
    with h5py.File(tmp_path / "STOCH_DTB_ELEMENTS_TEST" / "solutions.h5", "r") as f:
        for name in f.keys():
            dset = f[name]["all_states"]
            cols = list(dset.attrs["cols"])
            i_Y = [cols.index(spec) for spec in gas.species_names]

            for row in dset[()]:
                gas.TPY = 300.0, 101325.0, row[i_Y]
                for elem in ["H", "O", "N"]:
                    assert np.isclose(row[cols.index(f"Y_{elem}")], gas.elemental_mass_fraction(elem), rtol=1.0e-3)

            Y_H_all.append(dset[:, cols.index("Y_H")])

    # Mixing has changed the elements mass fractions
    assert not np.allclose(Y_H_all[0], Y_H_all[-1])
