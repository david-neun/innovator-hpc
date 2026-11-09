#!/bin/bash
# ---------------------------------------------------------------------
# Project 2 — tuning job for solve3d at the official 3D size, plus the
# direct rows lost when the natural ordering ran out of memory at m=60.
#     sbatch hpc_tune.sbatch
# ---------------------------------------------------------------------
#SBATCH --job-name=p2tune
#SBATCH --output=p2tune-%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --partition=compute

module load julia
mkdir -p results
echo "node: $(hostname)   started: $(date)"
# preconditioner variants at the official 3D size (BLAS 1 thread)
julia run_iterative.jl -d 3 --m 80 --rtol 1e-6 --precond ic5,ic10,ic20,amg,amgsa --out results/tune_hpc.csv
# the thread effect the handout describes: same solves with the node's BLAS threads
julia run_iterative.jl -d 3 --m 80 --rtol 1e-6 --precond ic,amg --blas 24 --out results/tune_blas_hpc.csv
# direct rows lost to the OOM: fill-reducing orderings only at 3D m=60 and m=80
julia run_direct.jl -d 3 --m 60,80 --orderings amd,metis,default --maxgb 10 --out results/direct_hpc.csv
echo "finished: $(date)"
