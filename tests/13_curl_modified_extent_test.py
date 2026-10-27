"""
Test 13: mixing extents of the CURL_MODIFIED model.

Creates a small H2 particle cloud (CURL_MODIFIED, mixing pairs drawn, not read).
Check: one random mixing extent is drawn for each mixing pair (Janicka et al. 1979, Pope 1982),
not one for all pairs of a time step.

Independent of the other tests.
"""
import os
import numpy as np
from ai_reacting_flows.stochastic_reactors_data_gen.particles_cloud import ParticlesCloud

import DummyComm as dc

# Mechanism file
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

inlet_1 = {"type": "cold_premixed", "nb_particles": 10, "fuel": "H2", "phi": 0.6, "T": 300.0, "P": 101325.0}
inlet_2 = {"type": "burnt_premixed", "nb_particles": 10, "fuel": "H2", "phi": 0.6, "T": 300.0, "P": 101325.0}

data_gen_parameters = {}
data_gen_parameters["mech_file"] = mech_file
data_gen_parameters["inlets"] = {"inlet_1": inlet_1, "inlet_2": inlet_2}
data_gen_parameters["results_folder_suffix"] = "CURL_EXTENT_TEST"
data_gen_parameters["dtb_file"] = "solutions.h5"
data_gen_parameters["build_ml_dtb"] = True
data_gen_parameters["time_step"] = 1.0e-5
data_gen_parameters["time_max"] = 3.0e-5
data_gen_parameters["automatic_termination"] = False
data_gen_parameters["mixing_model"] = "CURL_MODIFIED"
data_gen_parameters["mixing_time"] = 1.0e-5
data_gen_parameters["read_mixing"] = False
data_gen_parameters["T_threshold"] = 200.0
data_gen_parameters["pv_species"] = ["H2O"]
data_gen_parameters["fuel"] = ["H2"]
data_gen_parameters["calc_mean_traj"] = False
data_gen_parameters["ML_inference_flag"] = False


def test_one_extent_per_pair(tmp_path):
    os.chdir(tmp_path)
    os.mkdir("STOCH_DTB_CURL_EXTENT_TEST")   # Created by generate_stochastic_database in normal runs

    cloud = ParticlesCloud(data_gen_parameters, dc.DummyComm())

    for pairs, extents in zip(cloud.pairs_list, cloud.CURL_rate_list):
        assert len(extents) == len(pairs)
        assert np.all((extents >= 0.0) & (extents < 1.0))

    # Several pairs per time step, with different extents
    assert len(cloud.pairs_list[0]) > 1
    assert np.unique(cloud.CURL_rate_list[0]).size == len(cloud.pairs_list[0])
