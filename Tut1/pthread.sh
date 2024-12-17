#!/bin/bash
#SBATCH --job-name=pthread_addsub
#SBATCH --output=pthread_%j.log
#SBATCH --error=pthread_%j.log
echo "Running pthread job!"
echo "We are running on $(hostname)"
echo "Job started at $(date)"
# NOTE: LINE BELOW CHANGED TO RUN COND
srun --partition i7-7700 /usr/bin/time -vvv ./pthread_addsub
echo "Job ended at $(date)"