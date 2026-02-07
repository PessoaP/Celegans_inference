#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 0-168:00:00

#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1

#SBATCH -o /scratch/cylu1/logs/grid_likelihood/slurm.%j.out
#SBATCH -e /scratch/cylu1/logs/grid_likelihood/slurm.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cylu1@asu.edu
#SBATCH --export=NONE

# === Load Environment ===
module purge
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
source activate repopgraphing

# === Set Up Scratch Directory ===
export SCRATCH_RUN_DIR=/scratch/cylu1/grid_likelihood_$SLURM_JOB_ID
mkdir -p "$SCRATCH_RUN_DIR"
cp -r ~/Celegans_inference_grid/* "$SCRATCH_RUN_DIR"/
cd "$SCRATCH_RUN_DIR"

# === Ensure local imports work ===
export PYTHONPATH=$PWD:$PYTHONPATH

# === Inputs ===
# Path to the real data file we calculate likelihood for
REAL_DATA_PATH="real_data/Exp_1_live_lowpH.csv"

# === Run Script ===
python utils/grid_likelihood_calculation.py "$REAL_DATA_PATH"

# === Copy outputs back to a persistent location ===
# The script writes to: grid_likelihood_outputs/<basename>/grid_loglike.csv
BASENAME="$(basename "$REAL_DATA_PATH")"
BASENAME_NOEXT="${BASENAME%.*}"

OUTDIR=~/grid_likelihood_runs/$BASENAME_NOEXT/$SLURM_JOB_ID
mkdir -p "$OUTDIR"

# Copy the whole output folder for this dataset (includes resume CSV)
cp -rv "grid_likelihood_outputs/$BASENAME_NOEXT" "$OUTDIR"/

# Also keep a copy of the κ samples used (provenance)
cp -v "kappa_samples_Exp1_4gauss.npz" "$OUTDIR"/

