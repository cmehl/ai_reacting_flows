#!/bin/bash
# Baseline-equivalent processing (2 clusters, threshold 1e-10, single-step) of the LONG raw database.
module load GCC/13.3.0 GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.0 OpenMPI/5.0.3
export PATH=/work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.venv/bin:$PATH
export PYTHONPATH=/work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.claude/worktrees/cfd-rollout-dtb/src
cd /work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.claude/worktrees/cfd-rollout-dtb/scripts/NH3_H2_N2/STOCH_REDUCED_ROLLOUT_LONG
cp dtb_processing_single_k2.yaml dtb_processing.yaml
python dtb_processing.py > dtb_processing_single_k2.log 2>&1 || { echo "PROCESSING k2 FAILED"; exit 1; }
echo "PROCESSING k2 DONE"
