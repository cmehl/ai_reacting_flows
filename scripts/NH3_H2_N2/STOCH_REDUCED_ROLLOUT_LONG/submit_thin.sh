#!/bin/bash
# Uploads each thinned-tail experiment (E1, E1c, E2) to its own orion folder and submits its training job.
set -e
L=/work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.claude/worktrees/cfd-rollout-dtb/scripts/NH3_H2_N2/STOCH_REDUCED_ROLLOUT_LONG
O=/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/.idea
cd "$L"
declare -A DB=( [E1]=dtb_thin_rollout_k2_thr1e14 [E1c]=dtb_thin_single_k2_thr1e14 [E2]=dtb_thin_rollout_k1_thr1e14 )
for e in E1 E1c E2; do
  D=NH3_H2_N2_REDUCED_THIN_$e
  ssh -o BatchMode=yes kotlarcm@orion "mkdir -p $O/$D/.log $O/$D/STOCH_DTB_NH3_H2_N2_ROLLOUT_THIN"
  rsync -az --exclude '*.png' -e ssh STOCH_DTB_NH3_H2_N2_ROLLOUT_THIN/${DB[$e]} STOCH_DTB_NH3_H2_N2_ROLLOUT_THIN/dtb_params.yaml STOCH_DTB_NH3_H2_N2_ROLLOUT_THIN/STEC_A_noAR.yaml kotlarcm@orion:$O/$D/STOCH_DTB_NH3_H2_N2_ROLLOUT_THIN/
  rsync -az -e ssh STEC_A_noAR.yaml ../STOCH_REDUCED_ROLLOUT/ann_model_learning.py thin_configs/run_train_$e.sbatch kotlarcm@orion:$O/$D/
  rsync -az -e ssh thin_configs/networks_params_$e.yaml kotlarcm@orion:$O/$D/networks_params.yaml
  ssh -o BatchMode=yes kotlarcm@orion "cd $O/$D && sbatch run_train_$e.sbatch"
done
