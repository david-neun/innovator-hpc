#!/bin/bash
# =====================================================================
# Runs the whole LOCAL experiment set (laptop-sized problems).
# From inside projects/2:      bash run_all_local.sh
# Everything is appended to results/*.csv and echoed to results/local.log
# Expected wall time: ~10–20 minutes. Each line is an independent Julia
# process, so a failure in one does not stop the others.
# =====================================================================
mkdir -p results
J="julia --project=."
{
echo "===== started $(date) ====="
# ---- Question 1: orderings (natural/rcm auto-skipped when the band bound > 4 GB)
$J run_direct.jl    -d 2 --m 50,100,200,400,800  --maxgb 4
$J run_direct.jl    -d 3 --m 10,20,30,40         --maxgb 4
# ---- Question 2: preconditioners, at the official tolerances
$J run_iterative.jl -d 2 --m 50,100,200,400,800  --rtol 1e-8
$J run_iterative.jl -d 3 --m 10,20,30,40         --rtol 1e-6
# ---- Question 3: how the crossover moves with rtol (best candidates only)
$J run_iterative.jl -d 2 --m 200,400 --rtol 1e-4,1e-6,1e-10 --precond jacobi,ic,amg --out results/iterative_rtol.csv
$J run_iterative.jl -d 3 --m 30,40   --rtol 1e-4,1e-8,1e-10 --precond jacobi,ic,amg --out results/iterative_rtol.csv
echo "===== finished $(date) ====="
} 2>&1 | tee -a results/local.log
