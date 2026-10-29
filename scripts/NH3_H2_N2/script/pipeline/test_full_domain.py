# Usage: mpirun -n N python test_full_domain.py <train_folder>
# <train_folder> holds test_ann_vs_cvode.yaml (+ MODELS/CFD_DTB symlinks); single-step (nb_steps: 1).
import sys
from mpi4py import MPI
from ai_reacting_flows.ann_model_generation.cfd_full_domain_testing import FullDomainTester

FullDomainTester(run_folder=sys.argv[1]).run_mpi(MPI.COMM_WORLD)
