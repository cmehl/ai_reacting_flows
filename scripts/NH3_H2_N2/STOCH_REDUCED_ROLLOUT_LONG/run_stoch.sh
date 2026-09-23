#!/bin/bash
# Usage: bash run_stoch.sh <case_dir> <nb_procs> <log_file>
# Loads the ARF modules (equivalent of the ml_arf alias), points python at this worktree's src, runs the stochastic reactor.
module load GCC/13.3.0 GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.0 OpenMPI/5.0.3
export PATH=/work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.venv/bin:$PATH
export PYTHONPATH=/work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.claude/worktrees/cfd-rollout-dtb/src
cd "$1"
mpirun --use-hwthread-cpus -n "$2" python generate_stoch_dtb.py > "$3" 2>&1
