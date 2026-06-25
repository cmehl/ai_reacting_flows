from ai_reacting_flows.stochastic_reactors_data_gen.main import GenerateVariable_dt

from mpi4py import MPI
import os
import oyaml as yaml

# MPI world communicator
comm = MPI.COMM_WORLD

run_folder = os.getcwd()
with open(os.path.join(run_folder, "dtb_params.yaml"), "r") as file:
    data_gen_parameters = yaml.safe_load(file)


GenerateVariable_dt(data_gen_parameters, comm)
