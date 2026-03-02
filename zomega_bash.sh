#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 0-48:00:00

#SBATCH -p general
#SBATCH -q grp_spresse
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

#source activate NFdeconvolve

# python run_grid.py real_data/Exp_1_live_lowpH.csv \
#   --alpha 0.05 \
#   --mu 0:1 \
#   --omega 0 \
#   --rho 1 \
#   --kappa-samples-path kappa_samples_stratified_highpH.npz

# python run_grid.py real_data/Exp_2_reduced_live_lowpH.csv \
#   --alpha 0.05 \
#   --mu 0:1 \
#   --omega 0 \
#   --rho 0:1 \
#   --t-switch 0.0 \
#   --kappa-samples-path kappa_samples_stratified_highpH.npz

#   python run_grid.py real_data/Exp_3_time-limited_lowpH.csv \
#   --alpha 0.05 \
#   --mu 0:1 \
#   --omega 0 \
#   --rho 0:1 \
#   --t-switch 24.0 \
#   --kappa-samples-path kappa_samples_stratified_highpH.npz

# python run_grid.py real_data/Exp_4_live_highpH.csv \
#   --alpha 0.05 \
#   --mu 0:1 \
#   --omega 0 \
#   --rho 1 \
#   --kappa-samples-path kappa_samples_stratified_highpH.npz

#   python run_grid.py real_data/Exp_5_time-limited_highpH.csv \
#   --alpha 0.05 \
#   --mu 0:1 \
#   --omega 0 \
#   --rho 0:1 \
#   --t-switch 24.0 \
#   --kappa-samples-path kappa_samples_stratified_highpH.npz

#   python run_grid.py real_data/Exp_6_BB_time-limited_highpH.csv \
#   --alpha 0.05 \
#   --mu 0:1 \
#   --omega 0 \
#   --rho 0:1 \
#   --t-switch 24.0 \
#   --kappa-samples-path kappa_samples_stratified_highpH.npz

  python run_grid.py real_data/Exp_7_low-salinity_time-limited.csv \
  --alpha 0.04 \
  --mu 0:1 \
  --omega 0 \
  --rho 0:1 \
  --t-switch 24.0 \
  --kappa-samples-path kappa_samples_stratified_highpH.npz

  python run_grid.py real_data/Exp_8_BB_low-salinity_time-limited.csv \
  --alpha 0.04 \
  --mu 0:1 \
  --omega 0 \
  --rho 0:1 \
  --t-switch 24.0 \
  --kappa-samples-path kappa_samples_stratified_highpH.npz