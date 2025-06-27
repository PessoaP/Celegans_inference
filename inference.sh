#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 0-48:00:00

#SBATCH -p general                      
#SBATCH -q grp_spresse
#SBATCH -G a30:1

#SBATCH -o /scratch/cylu1/logs/adaptive_inference/slurm.%j.out
#SBATCH -e /scratch/cylu1/logs/adaptive_inference/slurm.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cylu1@asu.edu
#SBATCH --export=NONE

# === Load Environment ===
module purge
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
source activate repopgraphing

# === Set Up Scratch Directory ===
export SCRATCH_RUN_DIR=/scratch/cylu1/inference_run_adaptive_$SLURM_JOB_ID
mkdir -p "$SCRATCH_RUN_DIR"
cp -r ~/Celegans_inference/* "$SCRATCH_RUN_DIR"/
cp -r ~/Celegans_inference/real_data "$SCRATCH_RUN_DIR"/
cd "$SCRATCH_RUN_DIR"

# === Ensure local imports work ===
export PYTHONPATH=$PWD:$PYTHONPATH

# === Choose Dataset ===
REAL_DATA_FILE="real_data/Exp_1_Ecoli_population.csv"

# === Run Script ===
python mcmc_inference_real.py "$REAL_DATA_FILE"