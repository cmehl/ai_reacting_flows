"""
Test 11: mixing pairs with MPI.

Mixing is only performed by MPI rank 0, which also creates the results folder. Creates a particle
cloud as seen by rank 1 (CURL_MODIFIED, mixing pairs drawn, not read), before the results folder exists.
Check: rank 1 does not draw nor write the mixing pairs (pairs_list.pkl only written by rank 0).

Independent of the other tests.
"""
import os
from ai_reacting_flows.stochastic_reactors_data_gen.particles_cloud import ParticlesCloud

import DummyComm as dc

# Mechanism file
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

inlet_1 = {"type": "cold_premixed", "nb_particles": 10, "fuel": "H2", "phi": 0.6, "T": 300.0, "P": 101325.0}
inlet_2 = {"type": "burnt_premixed", "nb_particles": 10, "fuel": "H2", "phi": 0.6, "T": 300.0, "P": 101325.0}

data_gen_parameters = {}
data_gen_parameters["mech_file"] = mech_file
data_gen_parameters["inlets"] = {"inlet_1": inlet_1, "inlet_2": inlet_2}
data_gen_parameters["results_folder_suffix"] = "MPI_TEST"
data_gen_parameters["dtb_file"] = "solutions.h5"
data_gen_parameters["build_ml_dtb"] = True
data_gen_parameters["time_step"] = 1.0e-5
data_gen_parameters["time_max"] = 3.0e-5
data_gen_parameters["mixing_model"] = "CURL_MODIFIED"
data_gen_parameters["mixing_time"] = 1.0e-4
data_gen_parameters["read_mixing"] = False
data_gen_parameters["T_threshold"] = 200.0
data_gen_parameters["pv_species"] = ["H2O"]
data_gen_parameters["fuel"] = ["H2"]
data_gen_parameters["calc_mean_traj"] = False
data_gen_parameters["ML_inference_flag"] = False


class Rank1Comm(dc.DummyComm):
    """Communicator as seen by MPI rank 1."""
    def Get_rank(self):
        return 1


def test_mixing_pairs_rank_1(tmp_path):
    os.chdir(tmp_path)

    ParticlesCloud(data_gen_parameters, Rank1Comm())

    assert not os.path.exists(tmp_path / "STOCH_DTB_MPI_TEST" / "pairs_list.pkl")
