#!/bin/bash
#SBATCH --job-name=CANTERA_thin_E1c
#SBATCH --wckey=xfk89001
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --output=.log/train_slurm_%j.log

cd /ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/.idea/NH3_H2_N2_REDUCED_THIN_E1c
source ../../.venv/bin/activate
nvidia-smi -L
python -u ann_model_learning.py
