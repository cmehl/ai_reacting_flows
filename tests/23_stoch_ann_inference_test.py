"""
Test 23: stochastic reactors with ANN chemistry (ML_inference_flag).

Builds a small H2 learning database (N2 removed) and trains a model for one epoch, then runs
stochastic reactors (hot lean H2/air particles and burnt particles) with this model.
Check:
- one reaction step of the particles gives the same state as NNTesting.advance_state_NN (same
  NNInference functions), up to the clipping and normalization of Y by Cantera.
- with the hybrid mode and a negative tolerance, all particles are advanced with Cantera, as
  without ANN.
- an error is raised if the progress variable species differ from those of the model.
- a full run with ANN chemistry writes its solutions.

Independent of the other tests.
"""
import os
import copy
import h5py
import numpy as np
import cantera as ct
import oyaml as yaml
import pytest
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting
from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database
from ai_reacting_flows.stochastic_reactors_data_gen.particles_cloud import ParticlesCloud
from h2_learning_database import write_h2_learning_database, mech_file, dt

import DummyComm as dc

inlet_1 = {"type": "cold_premixed", "nb_particles": 10, "fuel": "H2", "phi": 0.6, "T": 1200.0, "P": 101325.0}
inlet_2 = {"type": "burnt_premixed", "nb_particles": 10, "fuel": "H2", "phi": 0.6, "T": 300.0, "P": 101325.0}

data_gen_parameters = {}
data_gen_parameters["mech_file"] = mech_file
data_gen_parameters["inlets"] = {"inlet_1": inlet_1, "inlet_2": inlet_2}
data_gen_parameters["results_folder_suffix"] = "ANN_TEST"
data_gen_parameters["dtb_file"] = "solutions.h5"
data_gen_parameters["build_ml_dtb"] = True
data_gen_parameters["time_step"] = dt
data_gen_parameters["time_max"] = 3 * dt
data_gen_parameters["automatic_termination"] = False
data_gen_parameters["mixing_model"] = "CURL_MODIFIED"
data_gen_parameters["mixing_time"] = 1.0e-4
data_gen_parameters["read_mixing"] = False
data_gen_parameters["T_threshold"] = 200.0
data_gen_parameters["pv_species"] = ["H2O"]
data_gen_parameters["fuel"] = ["H2"]
data_gen_parameters["calc_mean_traj"] = False
data_gen_parameters["ML_inference_flag"] = True
data_gen_parameters["ML_model_folder"] = "MODEL_TEST"
data_gen_parameters["ML_yk_renormalization"] = False
data_gen_parameters["ML_hybrid_ann_cvode"] = False
data_gen_parameters["ML_hybrid_ann_cvode_tol"] = 5.0e-05

testing_parameters = {"models_folder": "MODEL_TEST", "spec_to_plot": ["H2"], "yk_renormalization": False,
                      "hybrid_ann_cvode": False, "hybrid_ann_cvode_tol": 5.0e-05}


def particles_states(cloud):
    """Temperatures [K] and mass fractions [-] of the particles of cloud."""

    return np.array([part.T for part in cloud.particles_list]), np.array([part.Y for part in cloud.particles_list])


def test_stoch_ann_inference(tmp_path):
    os.chdir(tmp_path)
    write_h2_learning_database(tmp_path)
    NN_manager().train_all_clusters()
    os.mkdir("STOCH_DTB_ANN_TEST")   # Created by generate_stochastic_database in normal runs

    # One reaction step with the ANN, compared to NNTesting on each particle
    cloud = ParticlesCloud(data_gen_parameters, dc.DummyComm())
    T_old, Y_old = particles_states(cloud)
    progvar = [part.prog_var for part in cloud.particles_list]
    P = cloud.particles_list[0].P

    cloud._apply_reactions(dt)
    T_ann, Y_ann = particles_states(cloud)

    test = NNTesting(testing_parameters)
    gas = ct.Solution(mech_file)
    for i in range(len(T_old)):
        test.attribute_cluster(np.append(T_old[i], Y_old[i]), dt, progvar[i])
        T_new, Y_new = test.advance_state_NN(T_old[i], Y_old[i], P, dt)
        gas.TPY = T_new, P, Y_new

        assert np.isclose(T_ann[i], T_new, rtol=1.0e-12)
        assert np.allclose(Y_ann[i], gas.Y, rtol=1.0e-10, atol=1.0e-14)

    # Hybrid mode with a negative tolerance: all particles advanced with Cantera
    params = copy.deepcopy(data_gen_parameters)
    params["ML_hybrid_ann_cvode"] = True
    params["ML_hybrid_ann_cvode_tol"] = -1.0
    cloud_hybrid = ParticlesCloud(params, dc.DummyComm())
    cloud_hybrid._apply_reactions(dt)

    params["ML_inference_flag"] = False
    cloud_cvode = ParticlesCloud(params, dc.DummyComm())
    cloud_cvode._apply_reactions(dt)

    assert np.allclose(particles_states(cloud_hybrid)[1], particles_states(cloud_cvode)[1])

    # Progress variable species different from those of the model
    params = copy.deepcopy(data_gen_parameters)
    params["pv_species"] = ["H2O", "OH"]
    with pytest.raises(ValueError, match="pv_species"):
        ParticlesCloud(params, dc.DummyComm())

    # Full run with ANN chemistry
    with open("dtb_params.yaml", "w") as file:
        yaml.dump(data_gen_parameters, file, default_flow_style=False)
    generate_stochastic_database(dc.DummyComm())

    with h5py.File(tmp_path / "STOCH_DTB_ANN_TEST" / "solutions.h5", "r") as f:
        assert len(f.keys()) > 0
