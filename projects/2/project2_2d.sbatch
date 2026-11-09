#!/bin/bash
# ---------------------------------------------------------------------
# Project 2 — official 2D timing (same flags the grader will use)
# ---------------------------------------------------------------------
#SBATCH --job-name=project2_2d
#SBATCH --output=project2_2d-%j.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --partition=compute

module load julia

julia test.jl -d 2 -m 500 --rtol 1e-8
