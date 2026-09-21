"""Writes processing + training configs for the thinned-tail experiments (log threshold 1e-14 in ALL cases).

E1  : 2 clusters, PerSpeciesMLP 128-128-64 (= reference `perspecies` architecture), rollout
E1c : same as E1 but single-step (control isolating the rollout effect)
E2  : 0 cluster, PerSpeciesMLP 2x64, rollout
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUFFIX = "NH3_H2_N2_ROLLOUT_THIN"
REF_NETWORKS = "/work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.idea/NH3_H2_N2_REDUCED/MODELS/MODEL_NH3_H2_N2_REDUCED_perspecies/networks_params.yaml"

EXPERIMENTS = {
    "E1": dict(rollout=True, nb_clusters=2, arch="ref"),
    "E1c": dict(rollout=False, nb_clusters=2, arch="ref"),
    "E2": dict(rollout=True, nb_clusters=1, arch="2x64"),
}

PROC = """database_params:
  database_type: stoch
  dtb_folder_suffix: {suffix}
  dtb_file: solutions.h5
  database_name: {dbname}
  dt_var: false
  rollout: {rollout}
  fuel: NH3
  mech_file: ./STOCH_DTB_{suffix}/STEC_A_noAR.yaml
data_processing:
  log_transform_X: 1
  log_transform_Y: 1
  threshold: 1.0e-14
  T_threshold: 800.0
  output_omegas: true
  with_N_chemistry: true
data_clustering:
  clusterize_on: phys
  clustering_method: kmeans
  nb_clusters: {nb_clusters}
train_set_size: 0.75
"""

NET_2X64 = """database_path: STOCH_DTB_{suffix}/{dbname}
model_name_suffix: {model}
new_model_folder: true
networks_types:
- PerSpeciesMLP
networks_def:
- cluster0
clusters:
  cluster0:
    nb_units_in_layers_list:
    - 64
    - 64
    layers_activation_list:
    - tanh
    - tanh
    - Id
    layers_type:
    - dense
    - dense
    - dense
learning:
  initial_learning_rate: 0.001
  batch_size: 2048
  epochs_list:
  - 500
  decay_rate: 0.9991
"""

SBATCH = """#!/bin/bash
#SBATCH --job-name=CANTERA_thin_{name}
#SBATCH --wckey=xfk89001
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --output=.log/train_slurm_%j.log

cd /ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/.idea/NH3_H2_N2_REDUCED_THIN_{name}
source ../../.venv/bin/activate
nvidia-smi -L
python -u ann_model_learning.py
"""

os.makedirs(f"{HERE}/thin_configs", exist_ok=True)
for name, e in EXPERIMENTS.items():
    kind = "rollout" if e["rollout"] else "single"
    cl = f"k{e['nb_clusters']}"
    dbname = f"dtb_thin_{kind}_{cl}_thr1e14"
    model = f"NH3_H2_N2_REDUCED_THIN_{name}_{kind}_{cl}_thr1e14"
    with open(f"{HERE}/thin_configs/dtb_processing_{name}.yaml", "w") as f:
        f.write(PROC.format(suffix=SUFFIX, dbname=dbname, rollout=str(e["rollout"]).lower(), nb_clusters=e["nb_clusters"]))
    if e["arch"] == "ref":
        net = open(REF_NETWORKS).read().splitlines()
        out = []
        for line in net:
            if line.startswith("database_path:"):
                line = f"database_path: STOCH_DTB_{SUFFIX}/{dbname}"
            elif line.startswith("model_name_suffix:"):
                line = f"model_name_suffix: {model}"
            out.append(line)
        text = "\n".join(out) + "\n"
    else:
        text = NET_2X64.format(suffix=SUFFIX, dbname=dbname, model=model)
    with open(f"{HERE}/thin_configs/networks_params_{name}.yaml", "w") as f:
        f.write(text)
    with open(f"{HERE}/thin_configs/run_train_{name}.sbatch", "w") as f:
        f.write(SBATCH.format(name=name))
    print(name, dbname, model)
