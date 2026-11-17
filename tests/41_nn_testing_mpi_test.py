"""
Test 41: series of test cases (NNTesting.run_testing) shared between 2 MPI ranks.

Builds a small H2 learning database and trains a model for one epoch, then runs a series of 3 ignition cases on 1
process (no communicator) and with mpiexec on 2 ranks (rank 0: cases 0 and 2, rank 1: case 1).
Check:
- the summary written by rank 0 has the cases in the order of the series, with the same errors as on 1 process
  (CPU times and speed-ups excepted).
- each rank writes its own log file, with the steps of its cases.

Needs mpiexec (skipped otherwise). Independent of the other tests.
"""
import os
import sys
import glob
import shutil
import subprocess
import numpy as np
import oyaml as yaml
import pytest
import torch
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting
from h2_learning_database import write_h2_learning_database, dt, pressure

# 3 ignition cases on CPU with one torch thread, no validation set and no benchmark
testing_parameters = {
    "models_folder": "MODEL_TEST", "spec_to_plot": ["H2"], "yk_renormalization": 0, "hybrid_ann_cvode": 0,
    "hybrid_ann_cvode_tol": 5.0e-05, "device": "cpu", "torch_threads": 1,
    "0D_ignition": {"run": True, "plots": False, "phi": [0.5, 0.6, 0.7], "T0": [1200.0], "pressure": [pressure],
                    "dt": [dt], "max_simulation_time": 3 * dt, "T_eq_tolerance": 1.0, "stability_factor": 1.0},
    "0D_pyrolysis": {"run": False}, "1D_premixed": {"run": False}, "1D_diffusion": {"run": False},
    "validation_set": {"run": False}, "speedup_benchmark": {"run": False},
}

# CPU times [s] and speed-up: not the same between two runs
TIMING_KEYS = ["time_cvode", "time_ann", "speedup"]

testing_script = """from mpi4py import MPI
import oyaml as yaml
from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting
with open("nn_testing.yaml", "r") as file:
    params = yaml.safe_load(file)
NNTesting(params, MPI.COMM_WORLD).run_testing()
"""


def run_mpi(folder, script, n_ranks=2):
    """Run a python script with mpiexec in folder (as in test 19)."""
    with open(os.path.join(folder, "run.py"), "w") as file:
        file.write(script)

    # Allow more ranks than cores with Open MPI (ignored by other MPI implementations)
    env = dict(os.environ, OMPI_MCA_rmaps_base_oversubscribe="yes")
    result = subprocess.run(["mpiexec", "-n", str(n_ranks), sys.executable, "run.py"],
                            cwd=folder, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        pytest.fail(f"mpiexec failed (return code {result.returncode})\n"
                    f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}")


def read_summary(folder):
    with open(os.path.join(folder, "TESTING_MODEL_TEST", "summary_0D_ignition.yaml"), "r") as file:
        return yaml.safe_load(file)


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec not available")
def test_nn_testing_mpi(tmp_path):
    os.chdir(tmp_path)
    write_h2_learning_database(tmp_path)
    NN_manager().train_all_clusters()
    with open("nn_testing.yaml", "w") as file:
        yaml.dump(testing_parameters, file, default_flow_style=False)

    # 1 process
    torch_threads = torch.get_num_threads()
    NNTesting(testing_parameters).run_testing()
    torch.set_num_threads(torch_threads)   # Back to the default for the other tests
    summary_serial = read_summary(tmp_path)
    shutil.rmtree("TESTING_MODEL_TEST")

    # 2 ranks, in the same folder (same model)
    run_mpi(tmp_path, testing_script)
    summary_mpi = read_summary(tmp_path)

    # Cases in the order of the series, same errors
    assert summary_mpi["n_cases"] == 3 and summary_mpi["n_failed"] == 0
    assert [case["params"] for case in summary_mpi["cases"]] == [case["params"] for case in summary_serial["cases"]]
    for case_serial, case_mpi in zip(summary_serial["cases"], summary_mpi["cases"]):
        for key, value in case_serial["errors"].items():
            # Errors not defined (e.g. no ignition): also on 2 ranks
            if value is None:
                assert case_mpi["errors"][key] is None, key

            # Other errors, except the CPU times
            else:
                if key not in TIMING_KEYS:
                    assert np.isclose(case_mpi["errors"][key], value, rtol=1.0e-10), key

    # One log file per rank, with the steps of its cases (3 ANN steps per case)
    logs = sorted(glob.glob(os.path.join("TESTING_MODEL_TEST", "nn_testing_*.log")))
    assert len(logs) == 2 and logs[1].endswith("_rank1.log")
    nb_steps = [open(log).read().count("0D_STEP") for log in logs]
    assert nb_steps == [2 * 3, 3]
