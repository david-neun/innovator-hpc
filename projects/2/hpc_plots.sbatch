#!/bin/bash
#SBATCH --job-name=p2plots
#SBATCH --output=p2plots-%j.out
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=compute
module load julia
echo "node: $(hostname)   started: $(date)"
julia make_figures.jl
ls -la plot/
echo "finished: $(date)"
