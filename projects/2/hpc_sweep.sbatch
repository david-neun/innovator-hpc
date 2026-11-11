#!/bin/bash
# ---------------------------------------------------------------------
# Project 2 — the small-size sweep on the grading node, so that every
# number in the report comes from the same machine.
#     sbatch hpc_sweep.sbatch
# ---------------------------------------------------------------------
#SBATCH --job-name=p2sweep
#SBATCH --output=p2sweep-%j.out
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --partition=compute

module load julia
mkdir -p results
echo "node: $(hostname)   started: $(date)"
# Question 1 — orderings, small sizes (large ones are already in direct_hpc.csv)
julia run_direct.jl    -d 2 --m 50,100,200,400              --maxgb 10 --out results/direct_hpc.csv
julia run_direct.jl    -d 3 --m 10,20,30                    --maxgb 10 --out results/direct_hpc.csv
julia run_direct.jl    -d 2 --m 800 --orderings default,amd,metis       --out results/direct_hpc.csv
# Question 2 — preconditioners, small sizes, plus the smoothed-aggregation curve
julia run_iterative.jl -d 2 --m 50,100,200,400,800 --rtol 1e-8 --precond none,jacobi,ic,amg,amgsa --out results/iterative_hpc.csv
julia run_iterative.jl -d 3 --m 10,20,30           --rtol 1e-6 --precond none,jacobi,ic,amg,amgsa --out results/iterative_hpc.csv
julia run_iterative.jl -d 3 --m 40,50,60           --rtol 1e-6 --precond amgsa                    --out results/iterative_hpc.csv
julia run_iterative.jl -d 2 --m 500,1000           --rtol 1e-8 --precond amgsa                    --out results/iterative_hpc.csv
# 2D banded orderings at m=800 last (L ≈ 4 GB; if it is killed, nothing after it is lost)
julia run_direct.jl    -d 2 --m 800 --orderings natural,rcm  --maxgb 10 --out results/direct_hpc.csv
echo "finished: $(date)"
