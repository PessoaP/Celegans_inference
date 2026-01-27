#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 0-00:10:00

#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1

#SBATCH -o /scratch/cylu1/logs/capacity_day9/slurm.%j.out
#SBATCH -e /scratch/cylu1/logs/capacity_day9/slurm.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cylu1@asu.edu
#SBATCH --export=NONE

# === Load Environment ===
module purge
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
source activate repopgraphing

# === Set Up Scratch Directory ===
export SCRATCH_RUN_DIR=/scratch/cylu1/capacity_day9_repop_$SLURM_JOB_ID
mkdir -p "$SCRATCH_RUN_DIR"
cp -r ~/Celegans_inference_grid/* "$SCRATCH_RUN_DIR"/
cd "$SCRATCH_RUN_DIR"

# === Ensure local imports work ===
export PYTHONPATH=$PWD:$PYTHONPATH

# === Run Script ===
python utils/calibrate_capacity.py

# === Copy outputs back to a persistent location ===
OUTDIR=~/capacity_day9_repop/$SLURM_JOB_ID
mkdir -p "$OUTDIR"

# Copy the produced NPZ (and optionally logs/anything else)
cp -v capacity_day9_repop_precalibration.npz "$OUTDIR"/
