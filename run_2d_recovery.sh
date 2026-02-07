#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 0-72:00:00

#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1

#SBATCH -o /scratch/cylu1/logs/recovery_2d/slurm.%j.out
#SBATCH -e /scratch/cylu1/logs/recovery_2d/slurm.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cylu1@asu.edu
#SBATCH --export=NONE


# === Load Environment ===
module purge
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
source activate repopgraphing

# === Set Up Scratch Directory ===
export SCRATCH_RUN_DIR=/scratch/cylu1/recovery_2d_${SLURM_JOB_ID}
mkdir -p "$SCRATCH_RUN_DIR"
cp -r ~/Celegans_inference_grid/* "$SCRATCH_RUN_DIR"/
cd "$SCRATCH_RUN_DIR"

# === Ensure local imports work ===
export PYTHONPATH=$PWD:$PYTHONPATH

# === Inputs ===
DATA_PATH="synthetic_data/synthetic_data.csv"
KAPPA_NPZ="synthetic_data/kappa_samples_Exp1_4gauss.npz"

# One job, one GPU, no slicing
OUTDIR_REL="recovery_tests/grid_single_gpu"

# === Run Script ===
python run_2d_recovery.py \
  --data "$DATA_PATH" \
  --kappa "$KAPPA_NPZ" \
  --outdir "$OUTDIR_REL" \
  --device cuda:0 \
  --flush-every 50 \
  --t-switch 24.0 \
  --rho 0.1

# === Copy outputs back to a persistent location ===
OUTDIR=~/recovery_2d_runs/${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

cp -rv "$OUTDIR_REL" "$OUTDIR"/
cp -v  "$KAPPA_NPZ" "$OUTDIR"/
cp -v  "$DATA_PATH" "$OUTDIR"/
