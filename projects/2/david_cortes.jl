# =====================================================================
# Project 2 — Choosing a sparse solver
# David Cortes — CSC/DSE 589, Fall 2026
#
# solve2d(A, b; rtol) and solve3d(A, b; rtol) return x with
#       ‖b − A x‖₂ / ‖b‖₂ ≤ rtol
# for the sparse SPD diffusion systems built by problem.jl.
# the functions do not modify A or b.
#
# Method choice (from the experiments in run_direct.jl / run_iterative.jl /
# tune3d.jl, all measured on the grading node; see the report):
#   2D  → sparse Cholesky (CHOLMOD) with its default AMD ordering.
#         Fill grows only like n log n in 2D, so the factorization is the
#         fastest option at every size tried (0.3 s at m = 500), and a
#         direct solve meets any rtol for free.
#   3D  → conjugate gradient preconditioned by one V-cycle of smoothed-
#         aggregation algebraic multigrid. Fill in 3D grows like n^{4/3}
#         and factor time like n², so the direct method falls behind early;
#         multigrid is the only preconditioner whose iteration count does
#         not grow with m (17–25 iterations at m = 80 against 212 for
#         incomplete Cholesky), and smoothed aggregation has the cheapest
#         setup. Two measured refinements (tune3d.jl, tune3d_b.jl): one
#         forward Gauss–Seidel sweep before and one backward sweep after
#         coarse correction (a cheaper, still symmetric V-cycle), and
#         asking Krylov for a tolerance tighter than rtol so that the
#         true-residual check below passes on the first try (one CG call
#         instead of two). Incomplete Cholesky is kept as a variant.
#
# Packages required (registered): Krylov, AlgebraicMultigrid, LimitedLDLFactorizations
#
# my repo https://github.com/david-neun/innovator-hpc/
# here I include the code for:
# plotting, timing, and the solver functions themselves.
# =====================================================================

# test.jl loads this file inside its own module → load everything here.
using LinearAlgebra, SparseArrays
using Krylov                     # conjugate gradient
using AlgebraicMultigrid         # algebraic multigrid used as the preconditioner
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
const PRECOND_3D   = :amgsa  # :amgsa (smoothed-aggregation AMG), :amgrs (Ruge–Stüben AMG) or :ic (incomplete Cholesky)
const INNER_TOL    = 0.05    # Krylov's own stopping test uses the preconditioned residual; ask it for
                             # INNER_TOL × rtol so the true residual is already below rtol when it returns
const LIGHT_SMOOTH = true    # true: one forward Gauss–Seidel sweep before, one backward after (cheaper cycle,
                             # more iterations); false: the package's default symmetric sweeps
const IC_MEMORY    = 0       # extra entries per column kept by the incomplete factor when :ic is used

function solve3d(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64}; rtol = 1e-8)
    # The handout's advice: multithreaded BLAS level-1 calls interleaved with the
    # single-threaded sparse product can slow CG down. Restrict BLAS to one thread
    # for the duration of the solve, and put the global setting back afterwards.
    nthreads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        M = PRECOND_3D == :amgsa ? _amgsa_preconditioner(A) :
            PRECOND_3D == :amgrs ? _amgrs_preconditioner(A) : _ic_preconditioner(A)
        return _pcg(A, b, M, rtol)
    finally
        BLAS.set_num_threads(nthreads)
    end
end

# M⁻¹ r = one multigrid V-cycle built from the entries of A alone ("algebraic").
# κ(M⁻¹A) is essentially independent of m, so the iteration count stays ~10–20.
function _amgsa_preconditioner(A)
    if LIGHT_SMOOTH
        ml = smoothed_aggregation(A; presmoother  = GaussSeidel(ForwardSweep(), 1),
                                     postsmoother = GaussSeidel(BackwardSweep(), 1))
    else
        ml = smoothed_aggregation(A)
    end
    return aspreconditioner(ml)
end
_amgrs_preconditioner(A) = aspreconditioner(ruge_stuben(A))

# M = L D Lᵀ ≈ A from the lower triangle of A (A itself is untouched).
function _ic_preconditioner(A)
    P = lldl(tril(A); memory = IC_MEMORY)
    P.D .= abs.(P.D)                         # keep M SPD, as CG requires
    return P
end

function _pcg(A, b, M, rtol)
    # Krylov.cg stops when its PRECONDITIONED residual norm is ≤ (its rtol) × its own
    # starting residual norm — not the true residual the grader checks. We ask for
    # INNER_TOL × rtol, then check ‖b − A x‖₂/‖b‖₂ ourselves; if it is still not
    # small enough we warm-start from x asking for the reduction factor missing
    # (factor-2 margin). atol = 0 so the default absolute tolerance cannot
    # stop a preconditioned run early.
    itmax = 20_000
    x, stats = cg(A, b; M = M, ldiv = true, atol = 0.0, rtol = INNER_TOL * rtol, itmax = itmax)
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
