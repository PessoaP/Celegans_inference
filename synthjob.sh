#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4	       # Request 4 CPUs
#SBATCH --mem=256G     # Request 256 GB memory
#SBATCH -t 0-16:00:00  # maximum time = 16 hours

#SBATCH -p general     # max time able to request = 7 days
#SBATCH -q public      # for what queue you're waiting in
#SBATCH -G a100:1      # request 1 GPU

#SBATCH -o /scratch/<yourusername>/logs/slurm.%j.out 
#SBATCH -e /scratch/<yourusername>/logs/slurm.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<youremail@something.edu>
#SBATCH --export=NONE

# Load Mamba and activate environment
module purge
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest
source activate torchcuda

# Set up scratch directory for this run
export SCRATCH_RUN_DIR=/scratch/<yourusername>/Celegans_inference_run_$SLURM_JOB_ID
mkdir -p "$SCRATCH_RUN_DIR"
cp -r ~/Celegans_inference/* "$SCRATCH_RUN_DIR"

# Copy your code to scratch (optional for isolation)
cp -r ~/Celegans_inference/* "$SCRATCH_RUN_DIR"
cd "$SCRATCH_RUN_DIR"

# Ensure Python can still find your in-repo library
export PYTHONPATH=$PWD:$PYTHONPATH

# Generate synthetic data 
bash make_synthetic_data.sh

# Run the main script
python mcmc_example.py
