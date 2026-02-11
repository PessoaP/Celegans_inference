#!/bin/bash

#SBATCH --job-name=grid_like
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-168:00:00

#SBATCH -p public
#SBATCH -q public

#SBATCH --array=1-4
#SBATCH -G a30:1

#SBATCH -o /scratch/cylu1/logs/grid_likelihood/slurm.%A_%a.out
#SBATCH -e /scratch/cylu1/logs/grid_likelihood/slurm.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cylu1@asu.edu
#SBATCH --export=NONE

# === Load Environment ===
module purge
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
source activate repopgraphing

# === Set Up Scratch Directory ===
export SCRATCH_RUN_DIR=/scratch/cylu1/grid_likelihood_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$SCRATCH_RUN_DIR"
cp -r ~/Celegans_inference_grid/* "$SCRATCH_RUN_DIR"/
cd "$SCRATCH_RUN_DIR"

# === Ensure local imports work ===
export PYTHONPATH=$PWD:$PYTHONPATH

# keep CPU threading sane for a single process
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# === Inputs ===
REAL_DATA_PATH="real_data/Exp_1_live_lowpH.csv"

# === Run Script (one process, one GPU) ===
# Python should read SLURM_ARRAY_TASK_ID to decide which alpha chunk to run.
python utils/grid_likelihood_calculation_multi_GPU.py "$REAL_DATA_PATH"

# === Copy outputs back to a persistent location ===
BASENAME="$(basename "$REAL_DATA_PATH")"
BASENAME_NOEXT="${BASENAME%.*}"

OUTDIR=~/grid_likelihood_runs/$BASENAME_NOEXT/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
mkdir -p "$OUTDIR"

cp -rv "grid_likelihood_outputs/$BASENAME_NOEXT" "$OUTDIR"/
cp -v "kappa_samples_Exp1_4gauss.npz" "$OUTDIR"/
