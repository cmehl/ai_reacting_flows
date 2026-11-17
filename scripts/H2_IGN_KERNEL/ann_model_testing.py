from mpi4py import MPI
import oyaml as yaml
from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting

# Testing parameters (see README)
with open("nn_testing.yaml", "r") as file:
    testing_parameters = yaml.safe_load(file)

# Cases of each series shared between the MPI ranks: mpirun -n X python ann_model_testing.py
test = NNTesting(testing_parameters, MPI.COMM_WORLD)
test.run_testing()
