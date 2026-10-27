"""
Test 18: mixture fraction of stochastic particles for a fuel containing N.

Creates a small NH3 particle cloud with a pure fuel inlet and an air inlet.
Check: the mixture fraction is 1 for the fuel particles and 0 for the air particles.

Independent of the other tests.
"""
import os
import numpy as np
from ai_reacting_flows.stochastic_reactors_data_gen.particles_cloud import ParticlesCloud

import DummyComm as dc

# Mechanism file
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_nh3_noArHe.yaml")

inlet_1 = {"type": "fuel", "nb_particles": 2, "fuel": "NH3", "T": 300.0, "P": 101325.0}
inlet_2 = {"type": "air", "nb_particles": 2, "T": 300.0, "P": 101325.0}

data_gen_parameters = {}
data_gen_parameters["mech_file"] = mech_file
data_gen_parameters["inlets"] = {"inlet_1": inlet_1, "inlet_2": inlet_2}
data_gen_parameters["results_folder_suffix"] = "MIXFRAC_TEST"
data_gen_parameters["dtb_file"] = "solutions.h5"
data_gen_parameters["build_ml_dtb"] = False
data_gen_parameters["time_step"] = 1.0e-5
data_gen_parameters["time_max"] = 3.0e-5
data_gen_parameters["automatic_termination"] = False
data_gen_parameters["mixing_model"] = "CURL"
data_gen_parameters["mixing_time"] = 1.0e-4
data_gen_parameters["read_mixing"] = False
data_gen_parameters["T_threshold"] = 200.0
data_gen_parameters["pv_species"] = ["H2O"]
data_gen_parameters["fuel"] = ["NH3"]
data_gen_parameters["calc_mean_traj"] = False
data_gen_parameters["ML_inference_flag"] = False


def test_mixture_fraction_NH3(tmp_path):
    os.chdir(tmp_path)
    os.mkdir("STOCH_DTB_MIXFRAC_TEST")   # Created by generate_stochastic_database in normal runs

    cloud = ParticlesCloud(data_gen_parameters, dc.DummyComm())

    mix_frac = np.array([part.mix_frac for part in cloud.particles_list])
    assert np.allclose(mix_frac, [1.0, 1.0, 0.0, 0.0])
