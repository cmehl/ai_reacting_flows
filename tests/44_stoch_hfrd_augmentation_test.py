"""
Test 44: HFRD data augmentation of a stochastic reactors database (augment_dataset, see tools/hfrd.py).

Runs a small H2 stochastic reactors simulation without augmentation, then the same simulation (same mixing
pairs) with augmentation.
Check:
- the simulation is unchanged: all_states, and the first rows of X and Y of every iteration, are those of the
  run without augmentation;
- perturbed states are added (at most one per state) after them, with temperatures above 300 K and within
  1/8 of the temperature range of the database from their original state, mass fractions summing to 1 and
  mass fractions of elements in the range of the database;
- Y of a perturbed state is the state advanced with Cantera over the time step.

Independent of the other tests.
"""
import os
import shutil
import h5py
import numpy as np
import cantera as ct
import oyaml as yaml
from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database
from ai_reacting_flows.tools.utilities import get_elements_matrix

import DummyComm as dc

# Mechanism file
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

inlet_1 = {"type": "cold_premixed", "nb_particles": 10, "fuel": "H2", "phi": 0.6, "T": 1000.0, "P": 101325.0}
inlet_2 = {"type": "burnt_premixed", "nb_particles": 10, "fuel": "H2", "phi": 0.6, "T": 300.0, "P": 101325.0}

data_gen_parameters = {}
data_gen_parameters["mech_file"] = mech_file
data_gen_parameters["inlets"] = {"inlet_1": inlet_1, "inlet_2": inlet_2}
data_gen_parameters["results_folder_suffix"] = "HFRD_TEST"
data_gen_parameters["dtb_file"] = "solutions.h5"
data_gen_parameters["build_ml_dtb"] = True
data_gen_parameters["time_step"] = 1.0e-5
data_gen_parameters["time_max"] = 5.0e-5
data_gen_parameters["automatic_termination"] = False
data_gen_parameters["mixing_model"] = "CURL_MODIFIED"
data_gen_parameters["mixing_time"] = 1.0e-4
data_gen_parameters["read_mixing"] = False
data_gen_parameters["T_threshold"] = 200.0
data_gen_parameters["pv_species"] = ["H2O"]
data_gen_parameters["fuel"] = ["H2"]
data_gen_parameters["calc_mean_traj"] = False
data_gen_parameters["ML_inference_flag"] = False


def run(folder, params):
    """Stochastic reactors simulation in folder; returns the datasets of solutions.h5 {group: {dataset: array}}."""
    os.chdir(folder)
    with open("dtb_params.yaml", "w") as file:
        yaml.dump(params, file, default_flow_style=False)
    generate_stochastic_database(dc.DummyComm())

    with h5py.File(folder / "STOCH_DTB_HFRD_TEST" / "solutions.h5", "r") as f:
        return {grp: {dset: f[grp][dset][()] for dset in f[grp].keys()} for grp in f.keys()}


def test_stoch_hfrd_augmentation(tmp_path):

    # Without augmentation (mixing pairs drawn and written)
    reference = tmp_path / "reference"
    reference.mkdir()
    sol_ref = run(reference, dict(data_gen_parameters, augment_dataset=False))

    # With augmentation, same mixing pairs
    augmented = tmp_path / "augmented"
    augmented.mkdir()
    for name in ["pairs_list.pkl", "CURL_rate_list.pkl"]:
        shutil.copy(reference / "STOCH_DTB_HFRD_TEST" / name, augmented)
    sol_aug = run(augmented, dict(data_gen_parameters, augment_dataset=True, read_mixing=True))

    gas = ct.Solution(mech_file)
    n_spec = gas.n_species
    elements_matrix = get_elements_matrix(gas)

    # Ranges of the states X of the database
    X_all = np.concatenate([sol_ref[grp]["X"] for grp in sol_ref], axis=0)
    T_range = X_all[:, 0].max() - X_all[:, 0].min()
    Z_el = X_all[:, 2:2 + n_spec] @ elements_matrix.T
    Z_el_min = Z_el.min(axis=0) * 0.99
    Z_el_max = Z_el.max(axis=0) * 1.01

    nb_added = 0
    for grp in sol_ref:

        # Simulation unchanged
        np.testing.assert_allclose(sol_aug[grp]["all_states"], sol_ref[grp]["all_states"], rtol=1e-12)
        n = sol_ref[grp]["X"].shape[0]
        np.testing.assert_allclose(sol_aug[grp]["X"][:n], sol_ref[grp]["X"], rtol=1e-12)
        np.testing.assert_allclose(sol_aug[grp]["Y"][:n], sol_ref[grp]["Y"], rtol=1e-12)

        # Perturbed states
        X_new = sol_aug[grp]["X"][n:]
        Y_new = sol_aug[grp]["Y"][n:]
        assert X_new.shape[0] <= n
        assert Y_new.shape[0] == X_new.shape[0]
        nb_added += X_new.shape[0]

        assert np.all(X_new[:, 0] >= 300.0)
        np.testing.assert_allclose(X_new[:, 2:2 + n_spec].sum(axis=1), 1.0, rtol=1e-10)
        Z_el_new = X_new[:, 2:2 + n_spec] @ elements_matrix.T
        assert np.all(Z_el_new >= Z_el_min) and np.all(Z_el_new <= Z_el_max)

        # Temperature within 1/8 of the range of one of the original states
        dT = np.abs(X_new[:, 0][:, None] - sol_ref[grp]["X"][:, 0][None, :]).min(axis=1)
        assert np.all(dT <= T_range / 8 * (1 + 1e-12))

        # Y: state advanced with Cantera over the time step
        for x, y in zip(X_new, Y_new):
            gas.TPY = x[0], x[1], x[2:2 + n_spec]
            sim = ct.ReactorNet([ct.IdealGasConstPressureReactor(gas)])
            sim.advance(data_gen_parameters["time_step"])
            np.testing.assert_allclose(y[0], gas.T, rtol=1e-6)
            np.testing.assert_allclose(y[2:2 + n_spec], gas.Y, rtol=1e-4, atol=1e-10)

    assert nb_added > 0
