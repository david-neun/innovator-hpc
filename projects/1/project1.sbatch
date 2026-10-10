#!/bin/bash
#SBATCH --job-name=project1
#SBATCH --output=project1-%j.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
##SBATCH --exclusive

echo "host: $(hostname)  dir: $(pwd)  start: $(date)"
module load julia
which julia
julia david_cortes.jl -n 500
echo "end: $(date)"