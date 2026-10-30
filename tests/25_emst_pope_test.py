"""
Test 25: EMST mixing model of Subramaniam & Pope (emst_pope.py, mixing_model EMST_POPE).

Check:
- the tree of points on a line is the chain of the sorted points, and its edge coefficients are
  B = 2 min(W, 1 - W) (W: weight fraction on one side of the edge);
- ages: about half of the particles mixing in the stationary state (p0), ages within the durations of the
  states, switch of state when the remaining time ends;
- test case of emst_test.f (Ren, Subramaniam & Pope): 100 particles, 3 compositions, random weights and
  scales, one composition initially a delta function, random time steps: the variance function decays
  as exp(-s), the means are conserved;
- stochastic reactors with EMST_POPE (air and H2 inlets): EMST_age written in solutions.h5 with ages in
  their bounds, H element and enthalpy means conserved, mixture fraction variance decreasing;
  EMST_SIMPLIFIED still runs, an unknown mixing model raises an error.

Independent of the other tests.
"""
import os
import h5py
import numpy as np
import pandas as pd
import oyaml as yaml
import pytest
import ai_reacting_flows.stochastic_reactors_data_gen.emst_pope as ep
from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database

import DummyComm as dc

mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")


def test_tree_and_edge_coefficients():
    # Points on a line, in random order: the minimum spanning tree is the chain of the sorted points
    x = np.array([[3.0], [0.0], [2.0], [1.0]])
    edges = ep.tree_edges(x)
    assert {tuple(sorted(e)) for e in edges} == {(1, 3), (2, 3), (0, 2)}

    # Chain of 4 equal weights from its end: W = 1/4, 2/4, 3/4 on the child side
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    assert np.allclose(ep.edge_coefficients(edges, np.ones(4)), [0.5, 1.0, 0.5])


def test_ages():
    rng = np.random.default_rng(0)
    ages = ep.init_ages(20000, rng)

    p0 = (ep.Z1_LOWER + ep.Z1_UPPER) / (ep.Z0_LOWER + ep.Z0_UPPER + ep.Z1_LOWER + ep.Z1_UPPER)
    assert abs(np.mean(ages > 0.0) - p0) < 0.02
    assert ages.max() <= ep.Z1_UPPER and ages.min() >= -ep.Z0_UPPER

    # Remaining times decrease; states ending switch, with a new duration of the other state
    ages = np.array([0.1, -0.1, 0.005, -0.005])
    new_ages = ep.increment_ages(ages, 0.01, rng)
    assert np.allclose(new_ages[:2], [0.09, -0.09])
    assert -ep.Z0_UPPER <= new_ages[2] <= -ep.Z0_LOWER
    assert ep.Z1_LOWER <= new_ages[3] <= ep.Z1_UPPER


@pytest.mark.parametrize("seed", [0, 1])
def test_variance_decay(seed):
    # Test case of emst_test.f
    rng = np.random.default_rng(seed)
    n, nc = 100, 3
    fscale = rng.random(nc) + 0.1
    w = rng.random(n)
    f = rng.random((n, nc))
    f[1:, 0] = f[0, 0]
    ages = ep.init_ages(n, rng)

    vf0 = ep.variance_function(f / fscale, w)
    mean0 = w @ f

    s = 0.0
    while s < 2.0:
        ds = 1.0e-2 * rng.random()
        f, ages = ep.emst_mix(f, fscale, w, ages, ds, rng)
        s += ds

    assert abs(ep.variance_function(f / fscale, w) / vf0 / np.exp(-s) - 1.0) < 0.05
    assert np.allclose(w @ f, mean0, rtol=1.0e-12, atol=1.0e-14)


def run_stochastic_reactors(tmp_path, mixing_model):
    """Small H2 stochastic reactors run (20 particles of air and fuel, 30 iterations), all_states per iteration."""

    os.chdir(tmp_path)
    params = {"mech_file": mech_file, "dtb_file": "solutions.h5", "results_folder_suffix": "EMST_TEST",
              "inlets": {"inlet_1": {"type": "air", "nb_particles": 10, "T": 300.0, "P": 101325.0},
                         "inlet_2": {"type": "fuel", "nb_particles": 10, "fuel": "H2", "T": 300.0, "P": 101325.0}},
              "build_ml_dtb": False, "time_step": 1.0e-6, "time_max": 3.0e-5, "automatic_termination": False,
              "mixing_model": mixing_model, "mixing_time": 1.0e-5, "read_mixing": False, "T_threshold": 200.0,
              "pv_species": ["H2O"], "fuel": ["H2"], "calc_mean_traj": False, "ML_inference_flag": False}
    with open("dtb_params.yaml", "w") as file:
        yaml.dump(params, file, default_flow_style=False)

    generate_stochastic_database(dc.DummyComm())

    with h5py.File(tmp_path / "STOCH_DTB_EMST_TEST" / "solutions.h5", "r") as f:
        return [pd.DataFrame(f[name]["all_states"][()], columns=[str(c) for c in f[name]["all_states"].attrs["cols"]])
                for name in sorted(f.keys())]


def test_stochastic_reactors_emst_pope(tmp_path):
    states = run_stochastic_reactors(tmp_path, "EMST_POPE")

    # Ages written, within the durations of the states
    ages = np.concatenate([df["EMST_age"].values for df in states])
    assert np.all(np.isfinite(ages))
    assert ages.max() <= ep.Z1_UPPER and ages.min() >= -ep.Z0_UPPER

    # Means of the H element and of the temperature conserved (no reaction at 300 K)
    assert np.allclose([df["Y_H"].mean() for df in states], states[0]["Y_H"].mean(), rtol=1.0e-10)
    assert np.allclose([df["Temperature"].mean() for df in states], states[0]["Temperature"].mean(), rtol=1.0e-6)

    # Mixing: variance of the mixture fraction decreasing (dt / mixing_time = 0.1 per iteration)
    variances = [df["Mix_frac"].var() for df in states]
    assert variances[-1] < 0.2 * variances[0]
    assert np.all(np.diff(variances) <= 1.0e-12)


def test_stochastic_reactors_other_models(tmp_path):
    states = run_stochastic_reactors(tmp_path, "EMST_SIMPLIFIED")
    assert np.all(np.isnan(states[-1]["EMST_age"]))
    assert states[-1]["Mix_frac"].var() < states[0]["Mix_frac"].var()

    with pytest.raises(ValueError, match="Input error"):
        run_stochastic_reactors(tmp_path, "EMST")
