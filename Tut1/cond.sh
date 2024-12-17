#!/bin/bash
#SBATCH --job-name=cond
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1gb
#SBATCH --time=00:00:30
#SBATCH --output=cond_%j.log
#SBATCH --error=cond_%j.log
echo "Running cond job!"
echo "We are running on $(hostname)"
echo "Job started at $(date)"
# NOTE: LINE BELOW CHANGED TO RUN COND
srun ./cond
echo "Job ended at $(date)"
