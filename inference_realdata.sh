#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=264G
#SBATCH -t 0-96:00:00

#SBATCH -p public                      
#SBATCH -q public
#SBATCH -G a100:1

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
export SCRATCH_RUN_DIR=/scratch/cylu1/inference_corrected_realdata_$SLURM_JOB_ID
mkdir -p "$SCRATCH_RUN_DIR"
cp -r ~/Celegans_inference_real/* "$SCRATCH_RUN_DIR"/
cd "$SCRATCH_RUN_DIR"

# === Ensure local imports work ===
export PYTHONPATH=$PWD:$PYTHONPATH

# === Use Selected Dataset ===
DATA_FILE="real_data/Exp_4_Regular_Feed_pH7.3_data.csv"

# === Run Script ===
python inference_realdata.py "$DATA_FILE"