#!/bin/bash
# ---------------------------------------------------------------------
# Project 2
# ---------------------------------------------------------------------
#SBATCH --job-name=project2
#SBATCH --output=project2-%j.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=16G
#SBATCH --partition=compute

module load julia

julia test.jl -d 3 -m 40 --rtol 1e-6 # Make sure you change the flags to your desired system
