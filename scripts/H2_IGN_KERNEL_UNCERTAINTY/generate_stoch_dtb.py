from mpi4py import MPI
from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database

# MPI world communicator
comm = MPI.COMM_WORLD

# Call to database generation function
generate_stochastic_database(comm)