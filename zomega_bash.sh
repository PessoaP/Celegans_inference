#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 0-24:00:00

#SBATCH -p public
#SBATCH -q public
#SBATCH -G a30:1

#SBATCH -o /scratch/ppessoa/logs/zomega_2d/slurm.%j.out
#SBATCH -e /scratch/ppessoa/logs/zomega_2d/slurm.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ppessoa@asu.edu
#SBATCH --export=NONE


# === Load Environment ===
module purge
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
source activate repopgraphing

# === Set Up Scratch Directory ===
export SCRATCH_RUN_DIR=/scratch/ppessoa/recovery_2d_${SLURM_JOB_ID}
mkdir -p "$SCRATCH_RUN_DIR"
cp -r ~/Celegans_inference_grid/* "$SCRATCH_RUN_DIR"/
cd "$SCRATCH_RUN_DIR"

# === Ensure local imports work ===
export PYTHONPATH=$PWD:$PYTHONPATH

# python grid_zomega.py real_data/Exp_1_live_lowpH.csv

python grid_zomega.py real_data/Exp_2_reduced_live_lowpH.csv 

python grid_zomega.py real_data/Exp_3_time-limited_lowpH.csv \
  --t-switch 24.0 \
  --rho 0.1

python grid_zomega.py real_data/Exp_3_time-limited_lowpH.csv \
  --t-switch 24.0 \
  --rho 0.1
