"""
Test 10: stochastic reactors output when some particles are not yet active.

Runs a very small H2 stochastic reactors simulation (CURL_MODIFIED) where the first inlet is
activated after the end of the simulation, so only particles of the second inlet are active.
Check: all_states in solutions.h5 only contains particles of the second inlet, an error
is raised if no inlet is active at t = 0, and particles are not mixed before their activation.

Independent of the other tests.
"""
import os
import h5py
import numpy as np
import copy
import oyaml as yaml
import pytest
from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database
from ai_reacting_flows.stochastic_reactors_data_gen.particles_cloud import ParticlesCloud

import DummyComm as dc

# Mechanism file
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

# Inlet 1 activated at t = 1 s [s], after the end of the simulation
inlet_1 = {"type": "cold_premixed", "nb_particles": 10, "fuel": "H2", "phi": 0.6, "T": 300.0, "P": 101325.0,
           "activation_time": 1.0}
inlet_2 = {"type": "burnt_premixed", "nb_particles": 10, "fuel": "H2", "phi": 0.6, "T": 300.0, "P": 101325.0}

data_gen_parameters = {}
data_gen_parameters["mech_file"] = mech_file
data_gen_parameters["inlets"] = {"inlet_1": inlet_1, "inlet_2": inlet_2}
data_gen_parameters["results_folder_suffix"] = "INACTIVE_TEST"
data_gen_parameters["dtb_file"] = "solutions.h5"
data_gen_parameters["build_ml_dtb"] = True
data_gen_parameters["time_step"] = 1.0e-5
data_gen_parameters["time_max"] = 3.0e-5
data_gen_parameters["automatic_termination"] = False
data_gen_parameters["mixing_model"] = "CURL_MODIFIED"
data_gen_parameters["mixing_time"] = 1.0e-4
data_gen_parameters["read_mixing"] = False
data_gen_parameters["T_threshold"] = 200.0
data_gen_parameters["pv_species"] = ["H2O"]
data_gen_parameters["fuel"] = ["H2"]
data_gen_parameters["calc_mean_traj"] = False
data_gen_parameters["ML_inference_flag"] = False


def test_all_states_active_particles(tmp_path):
    os.chdir(tmp_path)
    with open("dtb_params.yaml", "w") as file:
        yaml.dump(data_gen_parameters, file, default_flow_style=False)

    generate_stochastic_database(dc.DummyComm())

    with h5py.File(tmp_path / "STOCH_DTB_INACTIVE_TEST" / "solutions.h5", "r") as f:
        for name in f.keys():
            dset = f[name]["all_states"]
            i_inlet = list(dset.attrs["cols"]).index("Inlet_number")
            assert dset.shape[0] == 10
            assert np.all(dset[:, i_inlet] == 2)


def test_no_inlet_active_at_start(tmp_path):
    os.chdir(tmp_path)

    # Both inlets activated after t = 0
    params = copy.deepcopy(data_gen_parameters)
    params["inlets"]["inlet_2"]["activation_time"] = 1.0e-5

    with pytest.raises(ValueError, match="activation_time = 0"):
        ParticlesCloud(params, dc.DummyComm())


def test_no_mixing_before_activation(tmp_path):
    os.chdir(tmp_path)
    os.mkdir("STOCH_DTB_INACTIVE_TEST")   # Created by generate_stochastic_database in normal runs

    # Inlet 1 (particles 0 to 9) activated at the start of the second iteration
    params = copy.deepcopy(data_gen_parameters)
    params["inlets"]["inlet_1"]["activation_time"] = 1.0e-5

    cloud = ParticlesCloud(params, dc.DummyComm())

    assert np.all(cloud.pairs_list[0] >= 10)
    assert np.any(cloud.pairs_list[1] < 10)
