#!/bin/bash
# P K3-ROLL tanh with a dt = 5e-6 s database (10x the 5e-7 s one), everything else as the 5e-7 s K3 (same states, same k-means 3,
# same per-species 2x64 tanh networks, same losses [none, inv_increment, inv_increment], rollout 3 steps):
#   build (3 nodes, 384 ranks) -> ARF processing (noclust, own folder because of the fixed-name dtb_processing.yaml) -> K3 split
#   (kmeans3_true predicted on X, eps recomputed) -> 3 trainings -> assemble (k-means artefacts of kmeans3_true) -> post107 tests with
#   time_step 5e-6 (1 step + 10 chained) -> merges.
set -e
R=/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/.idea3/MLP_CFD_TFM_AMR_ROLLOUT3
cd $R
K=k3dt5e6
TAG=persp_kmeans3roll_dt5e6
SUF=TFM_AMR_ROLLOUT3_dt5e6_$TAG
DTBDIR=CFD_DTB_TFM_AMR_ROLLOUT3_dt5e6
DB=$DTBDIR/dtb_rollout3_thresh1e11_kmeans3_T800_mc50000
test ! -e $DTBDIR
test ! -e $K
mkdir -p $K/.log $K/proc/.log
mv $R/staging_dt5e6/build_rollout_from_raw_dt5e6.py $R/staging_dt5e6/build_kmeans3_rollout_dt5e6.py $K/
rmdir $R/staging_dt5e6

# 1) build on 3 nodes
cat > $K/run_build_dt5e6.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=CANTERA_build_dt5e6
#SBATCH --wckey=xfk89001
#SBATCH --partition=standard
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=$R/$K/.log/%x_%j.log
cd $R
ml GCC/13.3.0 GCCcore/13.3.0 Python/3.12.3 CUDA/12.6.0 OpenMPI/5.0.3
source ../../.venv/bin/activate
mpirun -n 384 python3 -u $K/build_rollout_from_raw_dt5e6.py
cp $R/CFD_DTB_TFM_AMR_ROLLOUT3_dt5e7/dtb_params_cfd.yaml $R/$DTBDIR/
ls -la $R/$DTBDIR
EOF

# 2) ARF processing (noclust), in its own folder
ln -sfn ../../$DTBDIR $K/proc/$DTBDIR
cp dtb_processing_rollout.py $K/proc/
sed -e "s#dtb_folder_suffix: TFM_AMR_ROLLOUT3_dt5e7#dtb_folder_suffix: TFM_AMR_ROLLOUT3_dt5e6#" \
    -e "s#mech_file: ../../data/Giovanni/STEC_A_noAR.yaml#mech_file: /ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/data/Giovanni/STEC_A_noAR.yaml#" \
    dtb_processing.yaml > $K/proc/dtb_processing.yaml
grep -E 'dtb_folder_suffix|database_name|mech_file|nb_clusters' $K/proc/dtb_processing.yaml
sed -e "s#^cd .*#cd $R/$K/proc#" -e "s#source ../../.venv/bin/activate#source $R/../../.venv/bin/activate#" \
    -e "s#CANTERA_MLP_TFMRO_dtb#CANTERA_dtb_dt5e6#" run_dtb_processing.sbatch > $K/proc/run_dtb_processing.sbatch

# 3) K3 split
sed -e "s#CANTERA_build_kmeans3roll#CANTERA_k3split_dt5e6#" -e "s#/kmeans3roll\$#/$K#" -e "s#build_kmeans3_rollout.py#build_kmeans3_rollout_dt5e6.py#" \
    kmeans3roll/run_build_kmeans3_rollout.sbatch > $K/run_k3split.sbatch

# 4) trainings (same yaml as the 5e-7 s K3, only database / eps / suffix change)
for k in 0 1 2; do
  t=train_${TAG}_z$k
  mkdir -p $t
  ln -sfn ../MODELS $t/MODELS
  ln -sfn ../$DTBDIR $t/$DTBDIR
  python3 - train_persp_kmeans3roll_z$k/networks_params.yaml $t ${SUF}_z$k $DB/z$k $DB/eps_global.npy <<'PY'
import sys, oyaml as yaml
tpl, t, suf, db, eps = sys.argv[1:6]
y = yaml.safe_load(open(tpl))
y["model_name_suffix"] = suf
y["database_path"] = db
y["learning"]["eps_file"] = eps
yaml.safe_dump(y, open(f"{t}/networks_params.yaml", "w"), sort_keys=False)
print(t, y["database_path"], y["clusters"]["cluster0"]["nb_units_in_layers_list"], y["clusters"]["cluster0"]["layers_activation_list"],
      y["learning"]["loss_weighting"], y["learning"]["eps_file"])
PY
done

# 5) assemble + top training_data links
sed -e "s#TFM_AMR_ROLLOUT3_dt5e7_persp_kmeans3roll#$SUF#g" -e "s#CFD_DTB_TFM_AMR_ROLLOUT3_dt5e7/dtb_rollout3_thresh1e11_kmeans3_T800_mc50000#$DB#g" \
    kmeans3roll/assemble_persp_kmeans3roll.py > $K/assemble_$TAG.py
sed -e "s#CFD_DTB_TFM_AMR_ROLLOUT3_dt5e7/dtb_rollout3_thresh1e11_kmeans3_T800_mc50000#$DB#g" -e "s#MODEL_TFM_AMR_ROLLOUT3_dt5e7_persp_kmeans3roll#MODEL_$SUF#" \
    -e "s#test_persp_kmeans3roll\"#test_$TAG\"#" kmeans3roll/link_top_persp_kmeans3roll.py > $K/link_top_$TAG.py
grep -nE "^(OUT|ZONES|DB|RDB|DB =|MODEL|TEST) " $K/assemble_$TAG.py $K/link_top_$TAG.py | cut -c1-200
sed -e "s#persp_kmeans3roll#$TAG#g" -e "s#/kmeans3roll\$#/$K#" kmeans3roll/run_assemble_persp_kmeans3roll.sbatch > $K/run_assemble_$TAG.sbatch

# 6) tests post107 with time_step 5e-6
for tt in "" _chain10; do
  t=test_${TAG}$tt
  mkdir -p $t results/parts_${TAG}$tt
  ln -sfn ../MODELS $t/MODELS
  ln -sfn ../$DTBDIR $t/$DTBDIR
  sed -e "s#^models_folder: .*#models_folder: MODEL_$SUF#" -e "s#^output_file: .*#output_file: ../results/parts_${TAG}$tt/full_domain_${TAG}$tt.h5#" \
      -e "s#^time_step: .*#time_step: 5.0e-06#" test_persp_kmeans3roll$tt/test_ann_vs_cvode.yaml > $t/test_ann_vs_cvode.yaml
  grep -E 'time_step|nb_steps|models_folder|output_file' $t/test_ann_vs_cvode.yaml
done

S="sbatch --parsable"
B=$(cd $K && $S run_build_dt5e6.sbatch)
P=$(cd $K/proc && $S --dependency=afterok:$B run_dtb_processing.sbatch)
X=$(cd $K && $S --dependency=afterok:$P run_k3split.sbatch)
Z=""
MEM=(150G 170G 70G); TIME=(40:00:00 47:00:00 20:00:00)
for k in 1 0 2; do Z="$Z:$($S --dependency=afterok:$X --mem=${MEM[$k]} --time=${TIME[$k]} --job-name=CANTERA_MLP_TFMRO_train_${TAG}_z$k --export=ALL,TRAIN=train_${TAG}_z$k run_train.sbatch)"; done
A=$(cd $K && $S --dependency=afterok$Z run_assemble_$TAG.sbatch)
T1=$($S --dependency=afterok:$A --job-name=CANTERA_MLP_TFMRO_test_$TAG --export=ALL,TRAIN=test_$TAG run_test_full_domain.sbatch)
T2=$($S --dependency=afterok:$A --job-name=CANTERA_MLP_TFMRO_test_${TAG}_chain10 --export=ALL,TRAIN=test_${TAG}_chain10 run_test_full_domain.sbatch)
M1=$($S --dependency=afterok:$T1 --job-name=CANTERA_MLP_TFMRO_merge_$TAG --export=ALL,MERGE_TAG=$TAG run_merge.sbatch)
M2=$($S --dependency=afterok:$T2 --job-name=CANTERA_MLP_TFMRO_merge_${TAG}_chain10 --export=ALL,MERGE_TAG=${TAG}_chain10 run_merge.sbatch)
echo "build=$B processing=$P k3split=$X trainings$Z assemble=$A tests=$T1 $T2 merges=$M1 $M2"
