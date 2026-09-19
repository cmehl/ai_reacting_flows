#!/bin/bash
# Full-slice ANN-vs-CVODE test (test_ann_vs_cvode.yaml), run from this folder.
module load GCC/13.3.0 GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.0 OpenMPI/5.0.3
export PATH=/work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.venv/bin:$PATH
export PYTHONPATH=/work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.claude/worktrees/cfd-rollout-dtb/src
cd /work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.claude/worktrees/cfd-rollout-dtb/scripts/NH3_H2_N2/STOCH_REDUCED_ROLLOUT
python test_ann_vs_cvode.py > test_ann_vs_cvode.log 2>&1 && echo "TEST DONE" || echo "TEST FAILED"
