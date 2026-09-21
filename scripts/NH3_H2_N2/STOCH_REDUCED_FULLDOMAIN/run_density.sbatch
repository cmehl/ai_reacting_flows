#!/bin/bash
#SBATCH --job-name=CANTERA_density_check
#SBATCH --wckey=xfk89001
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=density_%j.log

cd /ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/.idea/NH3_H2_N2_FULLDOMAIN_ANALYSIS
source ../../.venv/bin/activate
python -u db_density_orion.py
