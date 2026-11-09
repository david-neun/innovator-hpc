#!/bin/bash
# ---------------------------------------------------------------------
# Project 2 — the deliberate out-of-memory data point for Question 1.
# The band bound predicts the natural ordering needs ~13 GB at 3D m=70 and
# ~26 GB at m=80; the job has 16 GB. Run it in ITS OWN job so that when it
# dies it does not take the other experiments with it.
#     sbatch hpc_oom.sbatch
# ---------------------------------------------------------------------
#SBATCH --job-name=p2oom
#SBATCH --output=p2oom-%j.out
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --partition=compute

module load julia
mkdir -p results
echo "node: $(hostname)   started: $(date)"
# --maxgb 100 disables the guard on purpose. m=70 first (may survive), then m=80 (should not).
julia run_direct.jl -d 3 --m 70 --orderings natural --maxgb 100 --out results/direct_hpc.csv
julia run_direct.jl -d 3 --m 80 --orderings natural --maxgb 100 --out results/direct_hpc.csv
echo "finished: $(date)   (if you are reading this, nothing ran out of memory)"
