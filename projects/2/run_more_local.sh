#!/bin/bash
# =====================================================================
# Second local batch: the OFFICIAL sizes (2D m=500, 3D m=80) plus 3D m=60,
# to choose between IC and AMG for solve3d. From inside projects/2:
#     bash run_more_local.sh
# Expected: ~5–10 minutes. Output in results/*_more.csv and results/local.log
# =====================================================================
J="julia --project=."
{
echo "===== more: started $(date) ====="
# direct at the official sizes (3D: metis first — it needs the least memory; default last)
$J run_direct.jl    -d 2 --m 500     --orderings default,amd,metis           --out results/direct_more.csv
$J run_direct.jl    -d 3 --m 60,80   --orderings metis,default --maxgb 4     --out results/direct_more.csv
# iterative at the official sizes; ic5/ic10 keep 5/10 extra entries per column in the incomplete factor
$J run_iterative.jl -d 2 --m 500     --rtol 1e-8 --precond ic,amg            --out results/iterative_more.csv
$J run_iterative.jl -d 3 --m 60,80   --rtol 1e-6 --precond jacobi,ic,ic5,ic10,amg --out results/iterative_more.csv
# the thread effect the handout talks about: same solves with all BLAS threads
$J run_iterative.jl -d 3 --m 80      --rtol 1e-6 --precond ic,amg --blas 10  --out results/iterative_blas.csv
echo "===== more: finished $(date) ====="
} 2>&1 | tee -a results/local.log
