"""
Test 33: particles below T_threshold (frozen chemistry) keep their own mass fractions with CURL_MODIFIED mixing.

Particles of an inlet below T_threshold never react, so their mass fractions are never replaced by a new array.
CURL_MODIFIED mixes in place (Y += ...): if the particles of an inlet shared one array of mass fractions, mixing
one of them would change all the others, without changing their enthalpy.

Small H2 stochastic reactors simulation: inlet 1 at 300 K, below T_threshold = 600 K, inlet 2 at 1200 K.
Check: the particles of inlet 1 do not share their mass fractions, the ones never selected for mixing keep
the composition of their inlet, and the inlet state kept for the renewal of the particles is unchanged.

Independent of the other tests.
"""
import os

import numpy as np

from ai_reacting_flows.stochastic_reactors_data_gen.particles_cloud import ParticlesCloud

import DummyComm as dc

# Mechanism file
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

# Inlet 1 (particles 0 to 29) below T_threshold, inlet 2 above
inlet_1 = {"type": "cold_premixed", "nb_particles": 30, "fuel": "H2", "phi": 0.4, "T": 300.0, "P": 101325.0}
inlet_2 = {"type": "cold_premixed", "nb_particles": 10, "fuel": "H2", "phi": 0.4, "T": 1200.0, "P": 101325.0}

data_gen_parameters = {}
data_gen_parameters["mech_file"] = mech_file
data_gen_parameters["inlets"] = {"inlet_1": inlet_1, "inlet_2": inlet_2}
data_gen_parameters["results_folder_suffix"] = "COLD_TEST"
data_gen_parameters["dtb_file"] = "solutions.h5"
data_gen_parameters["build_ml_dtb"] = False
data_gen_parameters["time_step"] = 5.0e-7
data_gen_parameters["time_max"] = 1.0e-5
data_gen_parameters["automatic_termination"] = False
data_gen_parameters["mixing_model"] = "CURL_MODIFIED"
data_gen_parameters["mixing_time"] = 4.0e-4
data_gen_parameters["read_mixing"] = False
data_gen_parameters["T_threshold"] = 600.0
data_gen_parameters["pv_species"] = ["H2O"]
data_gen_parameters["fuel"] = ["H2"]
data_gen_parameters["calc_mean_traj"] = False
data_gen_parameters["ML_inference_flag"] = False

# Number of iterations run
NB_ITERATIONS = 10


def test_cold_particles_keep_their_mass_fractions(tmp_path):
    os.chdir(tmp_path)
    os.mkdir("STOCH_DTB_COLD_TEST")   # Created by generate_stochastic_database in normal runs

    np.random.seed(0)
    cloud = ParticlesCloud(data_gen_parameters, dc.DummyComm())
    cold = cloud.particles_list[:inlet_1["nb_particles"]]
    Y_inlet = cold[0].state_ini[2:].copy()

    # Each particle has its own array of mass fractions
    assert not any(np.shares_memory(part.Y, cold[0].Y) for part in cold[1:])

    # Particles selected for mixing during the run
    mixed = set()
    for _ in range(NB_ITERATIONS):
        mixed.update(cloud.pairs_list[cloud.iteration].flatten().tolist())
        cloud.advance_time(cloud.dt[cloud.iteration])

    # Cold particles never mixed: neither reacted nor mixed, still the composition of their inlet
    never_mixed = [part for part in cold if part.num_part not in mixed]
    assert len(never_mixed) > 0
    for part in never_mixed:
        np.testing.assert_array_equal(part.Y, Y_inlet)
        np.testing.assert_array_equal(part.state[2:], Y_inlet)

    # Inlet state kept for the renewal of the particles (residence_time)
    np.testing.assert_array_equal(cold[0].state_ini[2:], Y_inlet)
