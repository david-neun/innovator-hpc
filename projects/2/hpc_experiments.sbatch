#!/bin/bash
# ---------------------------------------------------------------------
# Project 2 — experiments at the sizes the laptop cannot reach.
# Same resources as the grading job (48 CPUs, 16 GB).  Submit from projects/2:
#     sbatch hpc_experiments.sbatch
# Needs the packages in Julia's DEFAULT environment on the cluster:
#     julia setup_env.jl --global          (run once, on the login node — it only installs)
# ---------------------------------------------------------------------
#SBATCH --job-name=p2exp
#SBATCH --output=p2exp-%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --partition=compute

module load julia
mkdir -p results
echo "node: $(hostname)   started: $(date)"

# Question 1 at full size (natural/rcm are auto-skipped above 10 GB by the band bound)
julia run_direct.jl    -d 2 --m 500,1000       --maxgb 10 --out results/direct_hpc.csv
julia run_direct.jl    -d 3 --m 40,50,60,80    --maxgb 10 --out results/direct_hpc.csv
# Question 2 at full size, official tolerances
julia run_iterative.jl -d 2 --m 500,1000       --rtol 1e-8 --out results/iterative_hpc.csv
julia run_iterative.jl -d 3 --m 40,50,60,80    --rtol 1e-6 --out results/iterative_hpc.csv
# Question 3: tolerance sweep at the official sizes
julia run_iterative.jl -d 2 --m 500 --rtol 1e-4,1e-6,1e-10 --precond jacobi,ic,amg --out results/iterative_rtol_hpc.csv
julia run_iterative.jl -d 3 --m 80  --rtol 1e-4,1e-8,1e-10 --precond jacobi,ic,amg --out results/iterative_rtol_hpc.csv

echo "finished: $(date)"
