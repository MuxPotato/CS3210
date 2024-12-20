#!/bin/bash

## This is a Slurm job script for Lab 2: mm-omp.cpp

#SBATCH --job-name=lab2-mmomp-i7
#SBATCH --partition=i7-7700
#SBATCH --output=lab2_mmomp_%j-i7.slurmlog
#SBATCH --error=lab2_mmomp_%j-i7.slurmlog
#SBATCH --mail-type=NONE

# Check that two arguments were passed (matrix size and number of openmp threads)
if [ ! "$#" -eq 2 ]
then
  echo "Expecting 2 arguments (<matrix size> <num threads>), got $#"
  exit 1
fi

echo "Running job: $SLURM_JOB_NAME!"
echo "We are running on $(hostname)"
echo "Job started at $(date)"
echo "Arguments to your executable: $@"

# Compile your code in case you forgot
echo "Compiling..."
srun g++ -fopenmp -o mm-omp mm-omp.cpp
# Runs your script with the arguments you passed in
echo "Running..."
srun perf stat -e instructions,cycles,fp_arith_inst_retired.scalar_single ./mm-omp $@

echo "Job ended at $(date)"