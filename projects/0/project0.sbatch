#!/bin/bash
#SBATCH --job-name=project0
#SBATCH --output=project0-%j.out          
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=compute

module load julia                     

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

julia project0.jl
