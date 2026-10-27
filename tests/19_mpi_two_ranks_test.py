"""
Test 19: stochastic reactors and multi-dt generation with 2 MPI ranks.

Runs a small H2 stochastic reactors simulation (CURL_MODIFIED, some particles activated later)
on 1 rank (DummyComm), then with mpirun on 2 ranks reading the mixing pairs of the first run.
Then re-integrates the 2-rank database with 3 random time steps (GenerateVariable_dt) on 2 ranks.
Check:
- solutions.h5 is the same with 1 and 2 ranks (all_states, X and Y at every iteration), up to the
  solver tolerances (relative differences ~1e-6, mostly on HRR: the Cantera gas object of each rank
  has a different history). Errors in particles distribution or mixing would give O(1) differences.
- the multi-dt file has one group per rank, contains all states, and the ranks draw different time steps.

Needs mpirun (skipped otherwise). Independent of the other tests.
"""
import os
import sys
import shutil
import subprocess
import h5py
import numpy as np
import oyaml as yaml
import pytest
from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database

import DummyComm as dc

# Mechanism file
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

inlet_1 = {"type": "cold_premixed", "nb_particles": 10, "fuel": "H2", "phi": 0.6, "T": 1000.0, "P": 101325.0}
inlet_2 = {"type": "burnt_premixed", "nb_particles": 10, "fuel": "H2", "phi": 0.6, "T": 300.0, "P": 101325.0}
inlet_3 = {"type": "air", "nb_particles": 5, "T": 1000.0, "P": 101325.0, "activation_time": 2.0e-5}

data_gen_parameters = {}
data_gen_parameters["mech_file"] = mech_file
data_gen_parameters["inlets"] = {"inlet_1": inlet_1, "inlet_2": inlet_2, "inlet_3": inlet_3}
data_gen_parameters["results_folder_suffix"] = "MPI_TEST"
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

# Multi-dt parameters
data_gen_parameters["time_step_type"] = "random"
data_gen_parameters["time_step_range"] = [1.0e-7, 1.0e-6]
data_gen_parameters["nb_dt"] = 3
data_gen_parameters["new_file_name"] = "solutions_3dt.h5"

# Launchers run with mpirun
generation_script = """from mpi4py import MPI
from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database
generate_stochastic_database(MPI.COMM_WORLD)
"""
multi_dt_script = """from mpi4py import MPI
import oyaml as yaml
from ai_reacting_flows.databases_processing.generate_var_dt_dtb import GenerateVariable_dt
with open("dtb_params.yaml", "r") as file:
    params = yaml.safe_load(file)
GenerateVariable_dt("stoch", params, MPI.COMM_WORLD)
"""


def run_mpi(folder, script, n_ranks=2):
    """Run a python script with mpirun in folder."""
    with open(os.path.join(folder, "run.py"), "w") as file:
        file.write(script)
    result = subprocess.run(["mpirun", "--oversubscribe", "-n", str(n_ranks), sys.executable, "run.py"],
                            cwd=folder, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def read_solutions(file_name):
    """All datasets of solutions.h5, as a dict {group/dataset: array}."""
    with h5py.File(file_name, "r") as f:
        return {f"{grp}/{dset}": f[grp][dset][()] for grp in f.keys() for dset in f[grp].keys()}


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not available")
def test_two_ranks(tmp_path):

    # 1 rank (mixing pairs drawn and written)
    serial = tmp_path / "serial"
    serial.mkdir()
    os.chdir(serial)
    with open("dtb_params.yaml", "w") as file:
        yaml.dump(data_gen_parameters, file, default_flow_style=False)
    generate_stochastic_database(dc.DummyComm())

    # 2 ranks, with the mixing pairs of the 1-rank run
    parallel = tmp_path / "parallel"
    parallel.mkdir()
    for name in ["pairs_list.pkl", "CURL_rate_list.pkl"]:
        shutil.copy(serial / "STOCH_DTB_MPI_TEST" / name, parallel)
    params = dict(data_gen_parameters, read_mixing=True)
    with open(parallel / "dtb_params.yaml", "w") as file:
        yaml.dump(params, file, default_flow_style=False)
    run_mpi(parallel, generation_script)

    sol_serial = read_solutions(serial / "STOCH_DTB_MPI_TEST" / "solutions.h5")
    sol_parallel = read_solutions(parallel / "STOCH_DTB_MPI_TEST" / "solutions.h5")
    assert sol_serial.keys() == sol_parallel.keys()
    for name in sol_serial:
        assert np.allclose(sol_serial[name], sol_parallel[name], rtol=1.0e-5, atol=1.0e-12), name

    # Multi-dt generation on 2 ranks
    run_mpi(parallel, multi_dt_script)

    n_states = sum(sol_serial[name].shape[0] for name in sol_serial if name.endswith("/X"))
    with h5py.File(parallel / "STOCH_DTB_MPI_TEST" / "solutions_3dt.h5", "r") as f:
        assert list(f.keys()) == ["ITERATION_00000", "ITERATION_00001"]
        assert sum(f[grp]["X"].shape[0] for grp in f.keys()) == n_states
        assert not np.allclose(f["ITERATION_00000/DT"][0], f["ITERATION_00001/DT"][0])
