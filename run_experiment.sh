#!/bin/bash
#SBATCH --job-name=soccer_hmm
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# make sure logs folder exists
mkdir -p logs

source ~/.bashrc
conda activate soccer_hmm

# let Stan know how many threads to use
export STAN_NUM_THREADS=$SLURM_CPUS_PER_TASK

# run your experiment
python -m src.run_experiment \
  --no-load-chains \
  --config-path experiments/run_14/config.yaml