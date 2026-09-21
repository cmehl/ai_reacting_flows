#!/bin/bash
# Processes the thinned raw database for E1, E1c, E2 (fixed-name dtb_processing.yaml swapped in turn).
module load GCC/13.3.0 GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.0 OpenMPI/5.0.3
export PATH=/work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.venv/bin:$PATH
export PYTHONPATH=/work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.claude/worktrees/cfd-rollout-dtb/src
cd /work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.claude/worktrees/cfd-rollout-dtb/scripts/NH3_H2_N2/STOCH_REDUCED_ROLLOUT_LONG
for e in E1 E1c E2; do
  cp thin_configs/dtb_processing_$e.yaml dtb_processing.yaml
  python dtb_processing.py > dtb_processing_thin_$e.log 2>&1 || { echo "PROCESSING $e FAILED"; exit 1; }
  echo "PROCESSING $e DONE"
done
