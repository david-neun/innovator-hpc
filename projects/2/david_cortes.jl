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
# measured on the grading node; see the report):
#   2D  → sparse Cholesky (CHOLMOD) with its default AMD ordering.
#         Fill grows only like n log n in 2D, so the factorization is the
#         fastest option at every size tried (0.53 s at m = 500), and a
#         direct solve meets any rtol for free.
#   3D  → preconditioned conjugate gradient. Fill in 3D grows like n^{4/3}
#         and factor time like n², so the direct method falls behind the
#         iterative ones early. At the official size (m = 80, n = 512 000)
#         algebraic multigrid is the fastest preconditioner on the grading
#         node (12 iterations, 3.1 s) ahead of incomplete Cholesky (212
#         iterations, 3.8 s); the IC variant is kept below for comparison.
#
# Packages required (registered): Krylov, AlgebraicMultigrid, LimitedLDLFactorizations
# =====================================================================

# test.jl loads this file inside its own module → load everything here.
using LinearAlgebra, SparseArrays
using Krylov                     # conjugate gradient
using AlgebraicMultigrid         # Ruge–Stüben AMG used as the preconditioner
using LimitedLDLFactorizations   # incomplete Cholesky (limited-memory LDLᵀ), alternative

# ---------------------------------------------------------------------
# 2D : sparse Cholesky, P A Pᵀ = L Lᵀ with CHOLMOD's default (AMD) ordering
# ---------------------------------------------------------------------
function solve2d(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64}; rtol = 1e-8)
    F = cholesky(A)      # ordering + symbolic + numeric factorization (does not touch A)
    return F \ b         # P, forward solve with L, backward solve with Lᵀ, Pᵀ
end

# ---------------------------------------------------------------------
# 3D : preconditioned conjugate gradient
# ---------------------------------------------------------------------
const PRECOND_3D = :amg   # :amg (algebraic multigrid) or :ic (incomplete Cholesky)
const IC_MEMORY  = 0      # extra entries per column kept by the incomplete factor when :ic is used

function solve3d(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64}; rtol = 1e-8)
    # The handout's advice: multithreaded BLAS level-1 calls interleaved with the
    # single-threaded sparse product slow CG down. Restrict BLAS to one thread
    # for the duration of the solve, and put the global setting back afterwards.
    nthreads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        M = PRECOND_3D == :amg ? _amg_preconditioner(A) : _ic_preconditioner(A)
        return _pcg(A, b, M, rtol)
    finally
        BLAS.set_num_threads(nthreads)
    end
end

# M⁻¹ r = one V-cycle of classical (Ruge–Stüben) AMG built from the entries of A.
# κ(M⁻¹A) is essentially independent of m, so the iteration count stays ~10–15.
_amg_preconditioner(A) = aspreconditioner(ruge_stuben(A))

# M = L D Lᵀ ≈ A from the lower triangle of A (A itself is untouched).
function _ic_preconditioner(A)
    P = lldl(tril(A); memory = IC_MEMORY)
    P.D .= abs.(P.D)                         # keep M SPD, as CG requires
    return P
end

function _pcg(A, b, M, rtol)
    # Krylov.cg stops when its PRECONDITIONED residual norm is ≤ rtol × its own
    # starting residual norm — not the true residual the grader checks. So we
    # check ‖b − A x‖₂/‖b‖₂ ourselves and, if needed, warm-start from x asking
    # for the reduction factor still missing (with a factor-2 margin).
    # atol = 0: Krylov's default absolute tolerance (√eps) could otherwise stop
    # a preconditioned run early.
    itmax = 20_000
    x, stats = cg(A, b; M = M, ldiv = true, atol = 0.0, rtol = rtol, itmax = itmax)
    iters = stats.niter
    rr = _relres(A, b, x)
    restarts = 0
    while rr > rtol && restarts < 5 && iters < itmax
        restarts += 1
        x, stats = cg(A, b, x; M = M, ldiv = true, atol = 0.0, rtol = 0.5 * rtol / rr, itmax = itmax - iters)
        iters += stats.niter
        rr = _relres(A, b, x)
    end
    return x
end

_relres(A, b, x) = norm(b - A * x) / norm(b)
