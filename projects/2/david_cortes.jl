# =====================================================================
# Project 2 — Choosing a sparse solver
# David Cortes — CSC/DSE 589, Fall 2026
#
# solve2d(A, b; rtol) and solve3d(A, b; rtol) return x with
#       ‖b − A x‖₂ / ‖b‖₂ ≤ rtol
# for the sparse SPD diffusion systems built by problem.jl.
# Neither function modifies A or b.
#
# Method choice (from the experiments in run_direct.jl / run_iterative.jl,
# see the report):
#   2D  → sparse Cholesky (CHOLMOD) with its default AMD ordering.
#         Fill grows only like n log n in 2D, so the factorization is the
#         fastest option at every size tried, and a direct solve meets
#         any rtol for free.
#   3D  → preconditioned conjugate gradient. Fill in 3D grows like n^{4/3}
#         and factor time like n², so the direct method falls behind
#         around m ≈ 20–30. PCG with an incomplete Cholesky (limited-memory
#         LDLᵀ) preconditioner was the fastest iterative method measured.
#
# Packages required (registered): Krylov, LimitedLDLFactorizations
# =====================================================================

# test.jl loads this file inside its own module → load everything here.
using LinearAlgebra, SparseArrays
using Krylov                     # conjugate gradient
using LimitedLDLFactorizations   # incomplete Cholesky as limited-memory LDLᵀ

# ---------------------------------------------------------------------
# 2D : sparse Cholesky, P A Pᵀ = L Lᵀ with CHOLMOD's default (AMD) ordering
# ---------------------------------------------------------------------
function solve2d(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64}; rtol = 1e-8)
    F = cholesky(A)      # ordering + symbolic + numeric factorization (does not touch A)
    return F \ b         # P, forward solve with L, backward solve with Lᵀ, Pᵀ
end

# ---------------------------------------------------------------------
# 3D : conjugate gradient with an incomplete-Cholesky preconditioner
# ---------------------------------------------------------------------
const IC_MEMORY = 0      # extra entries per column kept in the incomplete factor (0 = IC(0)-like)

function solve3d(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64}; rtol = 1e-8)
    # The handout's advice: multithreaded BLAS level-1 calls interleaved with the
    # single-threaded sparse product slow CG down. Restrict BLAS to one thread
    # for the duration of the solve, and put the global setting back afterwards.
    nthreads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        return _pcg_ic(A, b, rtol)
    finally
        BLAS.set_num_threads(nthreads)
    end
end

function _pcg_ic(A, b, rtol)
    # Preconditioner  M = L D Lᵀ ≈ A  from the lower triangle of A (A itself is untouched).
    P = lldl(tril(A); memory = IC_MEMORY)
    P.D .= abs.(P.D)                         # keep M SPD, as CG requires

    # Krylov.cg stops when its PRECONDITIONED residual norm is ≤ rtol × its own
    # starting residual norm — not the true residual the grader checks. So we
    # check ‖b − A x‖₂/‖b‖₂ ourselves and, if needed, warm-start from x asking
    # for the reduction factor still missing (with a factor-2 margin).
    itmax = 20_000
    x, stats = cg(A, b; M = P, ldiv = true, atol = 0.0, rtol = rtol, itmax = itmax)
    iters = stats.niter
    rr = _relres(A, b, x)
    restarts = 0
    while rr > rtol && restarts < 5 && iters < itmax
        restarts += 1
        x, stats = cg(A, b, x; M = P, ldiv = true, atol = 0.0, rtol = 0.5 * rtol / rr, itmax = itmax - iters)
        iters += stats.niter
        rr = _relres(A, b, x)
    end
    return x
end

_relres(A, b, x) = norm(b - A * x) / norm(b)
