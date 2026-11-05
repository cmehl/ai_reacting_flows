"""
Test 29: renewal of the particles of the stochastic reactors (residence_time, renewal_start_time).

Runs small H2 stochastic reactors simulations (CURL_MODIFIED and EMST_POPE).
Check:
- a renewed particle has the state of a new particle of its inlet;
- particles are renewed at the expected rate, only after renewal_start_time and only if active;
- the X -> Y pairs of the database stay pure chemistry: Y is X advanced by dt with Cantera;
- particles do return to their inlet state in solutions.h5;
- the age of the particles adds up the time steps in a closed reactor, and has the expected steady mean and
  standard deviation with renewal; its evolution is plotted with the residence time;
- solutions.h5 is the same with 1 and 2 MPI ranks, with CURL_MODIFIED (needs mpiexec, skipped otherwise; with
  EMST_POPE, runs differ by O(1) even without renewal, from the solver tolerances).

Independent of the other tests.
"""
import os
import sys
import copy
import shutil
import subprocess
import h5py
import numpy as np
import oyaml as yaml
import cantera as ct
import pytest
from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database
from ai_reacting_flows.stochastic_reactors_data_gen.particles_cloud import ParticlesCloud
from ai_reacting_flows.stochastic_reactors_data_gen.post_processing import StochDatabase

import DummyComm as dc

# Mechanism file
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

dt = 1.0e-5   # [s]

inlet_1 = {"type": "cold_premixed", "nb_particles": 10, "fuel": "H2", "phi": 0.6, "T": 1200.0, "P": 101325.0}
inlet_2 = {"type": "burnt_premixed", "nb_particles": 10, "fuel": "H2", "phi": 0.6, "T": 300.0, "P": 101325.0}

data_gen_parameters = {}
data_gen_parameters["mech_file"] = mech_file
data_gen_parameters["inlets"] = {"inlet_1": inlet_1, "inlet_2": inlet_2}
data_gen_parameters["results_folder_suffix"] = "TEST"
data_gen_parameters["dtb_file"] = "solutions.h5"
data_gen_parameters["build_ml_dtb"] = True
data_gen_parameters["time_step"] = dt
data_gen_parameters["time_max"] = 1.0e-4
data_gen_parameters["automatic_termination"] = False
data_gen_parameters["mixing_model"] = "CURL_MODIFIED"
data_gen_parameters["mixing_time"] = 1.0e-4
data_gen_parameters["read_mixing"] = False
data_gen_parameters["T_threshold"] = 200.0
data_gen_parameters["pv_species"] = ["H2O"]
data_gen_parameters["fuel"] = ["H2"]
data_gen_parameters["calc_mean_traj"] = False
data_gen_parameters["ML_inference_flag"] = False
data_gen_parameters["residence_time"] = 2.0e-5

generation_script = """from mpi4py import MPI
from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database
generate_stochastic_database(MPI.COMM_WORLD)
"""


def make_cloud(tmp_path, **changes):
    """Cloud of particles of the case, with some parameters changed."""

    os.chdir(tmp_path)
    os.makedirs("STOCH_DTB_TEST", exist_ok=True)

    return ParticlesCloud(dict(copy.deepcopy(data_gen_parameters), **changes), dc.DummyComm())


def run_case(folder, **changes):
    """Stochastic reactors simulation in folder, on 1 rank."""

    folder.mkdir()
    os.chdir(folder)
    with open("dtb_params.yaml", "w") as file:
        yaml.dump(dict(copy.deepcopy(data_gen_parameters), **changes), file, default_flow_style=False)

    generate_stochastic_database(dc.DummyComm())


def read_solutions(file_name):
    """All datasets of solutions.h5, as a dict {group/dataset: (array, column names)}."""

    with h5py.File(file_name, "r") as f:
        return {f"{grp}/{dset}": (f[grp][dset][()], [str(c) for c in f[grp][dset].attrs["cols"]])
                for grp in f.keys() for dset in f[grp].keys()}


def test_renewed_state(tmp_path):
    cloud = make_cloud(tmp_path)
    new_particles = copy.deepcopy(cloud.particles_list)

    # Particles far from their inlet state: reaction and mixing
    cloud.residence_time = None
    for _ in range(5):
        cloud.advance_time(dt)
    assert not np.isclose(cloud.particles_list[0].T, new_particles[0].T)

    for part, new in zip(cloud.particles_list, new_particles):
        part.renew(cloud)
    cloud._update_vars()

    for part, new in zip(cloud.particles_list, new_particles):
        for name in ["T", "P", "hs", "Hs", "mass", "Yc_u", "Yc_eq", "prog_var", "equiv_ratio", "mix_frac", "hrr"]:
            assert np.isclose(getattr(part, name), getattr(new, name), rtol=1.0e-9, atol=1.0e-12), name
        for name in ["Y", "X", "mass_k", "state"]:
            assert np.allclose(getattr(part, name), getattr(new, name), rtol=1.0e-9, atol=1.0e-14), name


def test_renewal_rate(tmp_path, monkeypatch):
    from ai_reacting_flows.stochastic_reactors_data_gen.particle import Particle

    renewed = []
    monkeypatch.setattr(Particle, "renew", lambda self, parent: renewed.append(self.num_part))

    # Second inlet not active, first one with 10 particles
    inlets = copy.deepcopy(data_gen_parameters["inlets"])
    inlets["inlet_2"]["activation_time"] = 1.0
    cloud = make_cloud(tmp_path, inlets=inlets, renewal_start_time=1.0e-3)

    # Before renewal_start_time
    cloud._renew_particles()
    assert renewed == []

    # After: binomial law, 5 standard deviations
    cloud.time = 1.0e-3
    n_draws = 20000
    for _ in range(n_draws):
        cloud._renew_particles()

    p = 1.0 - np.exp(-dt / data_gen_parameters["residence_time"])
    n = 10 * n_draws
    assert abs(len(renewed) - n * p) < 5.0 * np.sqrt(n * p * (1.0 - p))

    # Only the particles of the active inlet, all of them
    assert set(renewed) == set(range(10))


@pytest.mark.parametrize("mixing_model", ["CURL_MODIFIED", "EMST_POPE"])
def test_database(tmp_path, mixing_model):
    run_case(tmp_path / "case", mixing_model=mixing_model)
    sol = read_solutions(tmp_path / "case" / "STOCH_DTB_TEST" / "solutions.h5")

    # X -> Y pairs: chemistry only, also for the particles renewed at the previous step
    gas = ct.Solution(mech_file)
    iterations = sorted({name.split("/")[0] for name in sol})
    for grp in iterations[1:]:
        X, Y = sol[f"{grp}/X"][0], sol[f"{grp}/Y"][0]
        for x, y in zip(X[::4], Y[::4]):
            gas.TPY = x[0], x[1], x[2:2 + gas.n_species]
            sim = ct.ReactorNet([ct.IdealGasConstPressureReactor(gas)])
            sim.advance(dt)
            assert np.isclose(gas.T, y[0], rtol=1.0e-5)
            assert np.allclose(gas.Y, y[2:2 + gas.n_species], rtol=1.0e-3, atol=1.0e-8)

    # Particles of the first inlet come back to their initial temperature (it is not kept without renewal)
    cols = sol[f"{iterations[0]}/all_states"][1]
    i_T, i_inlet = cols.index("Temperature"), cols.index("Inlet_number")
    n_back = 0
    for grp in iterations[1:]:
        states = sol[f"{grp}/all_states"][0]
        T_inlet_1 = states[states[:, i_inlet] == 1, i_T]
        n_back += np.count_nonzero(np.isclose(T_inlet_1, inlet_1["T"], rtol=1.0e-10))
    assert n_back > 0


def test_age_closed_reactor(tmp_path):
    run_case(tmp_path / "case", residence_time=None)
    sol = read_solutions(tmp_path / "case" / "STOCH_DTB_TEST" / "solutions.h5")

    # Age of all particles: time of the simulation
    for grp in {name.split("/")[0] for name in sol}:
        states, cols = sol[f"{grp}/all_states"]
        assert np.allclose(states[:, cols.index("Age")], states[:, cols.index("Time")], rtol=1.0e-10, atol=1.0e-15)


def test_age_with_renewal(tmp_path, monkeypatch):
    from ai_reacting_flows.stochastic_reactors_data_gen.particle import Particle

    cloud = make_cloud(tmp_path)

    # Only the ages are of interest here, the inlet states are not needed
    monkeypatch.setattr(Particle, "renew", lambda self, parent: setattr(self, "age", 0.0))

    tau = data_gen_parameters["residence_time"]
    ages = []
    for i in range(40000):
        cloud._increase_ages(dt)
        cloud._renew_particles()

        # After the transient
        if i > 100:
            ages.append([part.age for part in cloud.particles_list])
    ages = np.array(ages)

    # Geometric law, after the renewal step
    p = 1.0 - np.exp(-dt / tau)
    mean = dt * (1.0 - p) / p
    std = dt * np.sqrt(1.0 - p) / p
    assert np.isclose(ages.mean(), mean, rtol=0.03)
    assert np.isclose(ages.std(), std, rtol=0.03)


def test_age_plot(tmp_path):
    run_case(tmp_path / "case")
    database = StochDatabase("STOCH_DTB_TEST", "postpro", with_traj=False)
    database.plot_age_time()
    assert os.path.isfile("postpro/age_time_plot.png")


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec not available")
def test_two_ranks(tmp_path):
    run_case(tmp_path / "serial")

    # 2 ranks, with the mixing pairs of the 1-rank run
    parallel = tmp_path / "parallel"
    parallel.mkdir()
    for name in ["pairs_list.pkl", "CURL_rate_list.pkl"]:
        shutil.copy(tmp_path / "serial" / "STOCH_DTB_TEST" / name, parallel)
    params = dict(copy.deepcopy(data_gen_parameters), read_mixing=True)
    with open(parallel / "dtb_params.yaml", "w") as file:
        yaml.dump(params, file, default_flow_style=False)
    with open(parallel / "run.py", "w") as file:
        file.write(generation_script)

    # Allow more ranks than cores with Open MPI (ignored by other MPI implementations)
    env = dict(os.environ, OMPI_MCA_rmaps_base_oversubscribe="yes")
    result = subprocess.run(["mpiexec", "-n", "2", sys.executable, "run.py"], cwd=parallel, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]

    sol_serial = read_solutions(tmp_path / "serial" / "STOCH_DTB_TEST" / "solutions.h5")
    sol_parallel = read_solutions(parallel / "STOCH_DTB_TEST" / "solutions.h5")
    assert sol_serial.keys() == sol_parallel.keys()
    for name in sol_serial:
        (x, cols), (y, _) = sol_serial[name], sol_parallel[name]

        # HRR of nearly unreacted particles has no relative accuracy: absolute tolerance from the largest HRR
        hrr = cols.index("HRR")
        assert np.allclose(np.delete(x, hrr, axis=1), np.delete(y, hrr, axis=1), rtol=1.0e-5, atol=1.0e-12, equal_nan=True), name
        assert np.allclose(x[:, hrr], y[:, hrr], rtol=1.0e-5, atol=1.0e-5 * np.max(np.abs(x[:, hrr]))), name
