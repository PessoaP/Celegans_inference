#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 0-10:00:00

#SBATCH -p general                      
#SBATCH -q public
#SBATCH -G 1

#SBATCH -o /scratch/cylu1/logs/synthetic_inference/slurm.%j.out
#SBATCH -e /scratch/cylu1/logs/synthetic_inference/slurm.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cylu1@asu.edu
#SBATCH --export=NONE

# === Load Environment ===
module purge
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
source activate repopgraphing

# === Set Up Scratch Directory ===
export SCRATCH_RUN_DIR=/scratch/cylu1/parameter_sweep_likelihoods_$SLURM_JOB_ID
mkdir -p "$SCRATCH_RUN_DIR"
cp -r ~/Celegans_inference_synthetic/* "$SCRATCH_RUN_DIR"/
cd "$SCRATCH_RUN_DIR"

# === Ensure local imports work ===
export PYTHONPATH=$PWD:$PYTHONPATH

# === Use Synthetic Dataset ===
SYNTH_DATA_FILE="synthetic_data/synthetic_data.csv"

# === Run Script ===
python 	test_parameter_likelihoods.py "$SYNTH_DATA_FILE"