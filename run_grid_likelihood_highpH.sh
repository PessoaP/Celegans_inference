#!/bin/bash

#SBATCH --job-name=grid_like_highpH
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 0-48:00:00

#SBATCH -p general
#SBATCH -q grp_spresse
#SBATCH -G a30:1

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
export SCRATCH_RUN_DIR=/scratch/cylu1/grid_likelihood_${SLURM_JOB_ID}
mkdir -p "$SCRATCH_RUN_DIR"
cp -r ~/Celegans_inference_grid/* "$SCRATCH_RUN_DIR"/
cd "$SCRATCH_RUN_DIR"

# === Ensure local imports work ===
export PYTHONPATH=$PWD:$PYTHONPATH

# === Inputs ===
REAL_DATA_PATH="real_data/Exp_4_live_highpH.csv"

# === Run Script (one process, one GPU) ===
python grid_zomega_highpH.py "$REAL_DATA_PATH" \

# === Copy outputs back to a persistent location ===
BASENAME="$(basename "$REAL_DATA_PATH")"
BASENAME_NOEXT="${BASENAME%.*}"

OUTDIR=~/grid_likelihood_runs/$BASENAME_NOEXT/$SLURM_JOB_ID
mkdir -p "$OUTDIR"

cp -rv "grid_likelihood_outputs/$BASENAME_NOEXT" "$OUTDIR"/
cp -v "kappa_samples_stratified_highpH.npz" "$OUTDIR"/
cp -v "$REAL_DATA_PATH" "$OUTDIR"/