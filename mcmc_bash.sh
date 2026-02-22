#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 0-167:00:00

#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1

#SBATCH -o /scratch/ppessoa/logs/MCMC/slurm.%j.out
#SBATCH -e /scratch/ppessoa/logs/MCMC/slurm.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ppessoa@asu.edu
#SBATCH --export=NONE

# === Load Environment ===
module purge
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
source activate repopgraphing

# === Set Up Scratch Directory ===
export SCRATCH_RUN_DIR=/scratch/ppessoa/mcmc_$SLURM_JOB_ID
mkdir -p "$SCRATCH_RUN_DIR"
cp -r ~/Celegans_inference_mcmc/* "$SCRATCH_RUN_DIR"/
cd "$SCRATCH_RUN_DIR"

# === Ensure local imports work ===
export PYTHONPATH=$PWD:$PYTHONPATH

# === Run Script ===
python mcmc_script.py