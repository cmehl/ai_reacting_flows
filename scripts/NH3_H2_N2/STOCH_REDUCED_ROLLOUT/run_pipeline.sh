#!/bin/bash
# dtb_processing -> rollout training -> full-slice masked test, sequentially. Run from anywhere.
module load GCC/13.3.0 GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.0 OpenMPI/5.0.3
export PATH=/work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.venv/bin:$PATH
export PYTHONPATH=/work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.claude/worktrees/cfd-rollout-dtb/src
cd /work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.claude/worktrees/cfd-rollout-dtb/scripts/NH3_H2_N2/STOCH_REDUCED_ROLLOUT
python dtb_processing.py > dtb_processing.log 2>&1 || { echo "PROCESSING FAILED"; exit 1; }
echo "PROCESSING DONE"
python ann_model_learning.py > ann_model_learning.log 2>&1 || { echo "TRAINING FAILED"; exit 1; }
echo "TRAINING DONE"
python test_ann_vs_cvode.py > test_ann_vs_cvode.log 2>&1 || { echo "TESTING FAILED"; exit 1; }
echo "TESTING DONE"
