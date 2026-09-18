#!/bin/bash
# ---------------------------------------------------------------------
# Project 1 
# ---------------------------------------------------------------------
#SBATCH --job-name=project1
#SBATCH --output=project1-%j.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --exclusive        

module load julia

#julia test.jl -n 500 # Make sure you change the flag to your desired dimension
julia david_cortes.jl -n 500 # Make sure you change the flag to your desired dimension

