#!/bin/bash
#SBATCH --job-name=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1gb
#SBATCH --time=00:00:30
#SBATCH --output=basic_%j.log
#SBATCH --error=basic_%j.log
echo "Running basic job!"
echo "We are running on $(hostname)"
echo "Job started at $(date)"
# Sleep "job", we use "srun" for better accounting
srun sleep 5;
# This is useful to know (in the logs) when the job ends
echo "Job ended at $(date)"