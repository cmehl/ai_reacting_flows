#!/bin/bash
module load GCC/13.3.0 GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.0 OpenMPI/5.0.3
export PATH=/work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.venv/bin:$PATH
cd /work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.claude/worktrees/cfd-rollout-dtb/scripts/NH3_H2_N2/STOCH_REDUCED_FULLDOMAIN
python -u analyze_catastrophic_cells.py > analyze_catastrophic_cells.log 2>&1 && echo "CATA DONE" || echo "CATA FAILED"
