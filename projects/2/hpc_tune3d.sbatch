#!/bin/bash
#SBATCH --job-name=p2tune3d
#SBATCH --output=p2tune3d-%j.out
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --partition=compute
module load julia
echo "node: $(hostname)   started: $(date)"
julia tune3d.jl
echo "finished: $(date)"
