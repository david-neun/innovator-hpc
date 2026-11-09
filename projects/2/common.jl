# =====================================================================
# Shared helpers for the Project 2 experiments
# (included by run_direct.jl and run_iterative.jl — not by david_cortes.jl)
# =====================================================================
using LinearAlgebra, SparseArrays, Printf
using AMD, SymRCM, Metis                                  # orderings
using Krylov, AlgebraicMultigrid, LimitedLDLFactorizations # CG + preconditioners

include(joinpath(@__DIR__, "problem.jl"))

# ---------------- command-line helpers ----------------
function getarg(flag, default)
    i = findfirst(==(flag), ARGS)
    return i === nothing ? default : parse(typeof(default), ARGS[i+1])
end
function getstr(flag, default::String)
    i = findfirst(==(flag), ARGS)
    return i === nothing ? default : ARGS[i+1]
end
getlist(flag, default)   = String.(split(getstr(flag, join(default, ",")), ","))
getints(flag, default)   = parse.(Int, getlist(flag, default))
getfloats(flag, default) = parse.(Float64, getlist(flag, default))

# ---------------- the system ----------------
build(d, m)     = problem(m; d)
relres(A, b, x) = norm(b - A * x) / norm(b)      # the grader's quantity

# ---------------- CSV logging: one row per run, header written once ----------------
function logrow(path, header, row)
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    newfile = !isfile(path)
    open(path, "a") do io
        newfile && println(io, join(header, ","))
        println(io, join(string.(row), ","))
    end
end

# ---------------- memory guard for the banded orderings ----------------
# Band bound (study notes §3.4): nnz(L) ≲ n·β with β = m (2D), m² (3D); 8 bytes/entry.
band_bytes(d, m) = d == 2 ? 8.0 * m^3 : 8.0 * m^5

# ---------------- orderings (Question 1) ----------------
const ALL_ORDERINGS = ["default", "natural", "rcm", "amd", "metis"]

function ordering_perm(name::String, A)
    n = size(A, 1)
    name == "default" && return nothing                        # let CHOLMOD choose
    name == "natural" && return collect(1:n)                   # the grid's own numbering
    name == "rcm"     && return Int.(symrcm(A))                # reverse Cuthill–McKee
    name == "amd"     && return Int.(amd(A))                   # approximate minimum degree
    name == "metis"   && return Int.(Metis.permutation(A)[1])  # nested dissection
    error("unknown ordering: $name")
end

# nonzeros of the Cholesky factor (nnz(F) exists for CHOLMOD factors; fallback just in case)
factor_nnz(F) = try nnz(F) catch; nnz(sparse(F.L)) end

# ---------------- preconditioners (Question 2) ----------------
const ALL_PRECONDS = ["none", "jacobi", "ic", "amg"]   # also available: ic5, ic10, ic20, amgsa

# Returns (M, ldiv) for Krylov.cg:
#   ldiv = false  →  Krylov applies M with mul!,  so M must be the INVERSE operator  (Jacobi: 1/diag)
#   ldiv = true   →  Krylov applies M with ldiv!, so M is a factorization-like object (IC, AMG)
function make_precond(name::String, A)
    if name == "none"
        return nothing, false
    elseif name == "jacobi"
        return Diagonal(1.0 ./ diag(A)), false
    elseif startswith(name, "ic")           # "ic" = no extra fill, "ic5" = 5 extra entries per column
        mem = name == "ic" ? 0 : parse(Int, name[3:end])
        P = lldl(tril(A); memory = mem)      # limited-memory incomplete LDLᵀ (Lin–Moré)
        P.D .= abs.(P.D)                     # safety: keep the preconditioner SPD
        return P, true
    elseif name == "amg"
        return aspreconditioner(ruge_stuben(A)), true   # classical (Ruge–Stüben) algebraic multigrid
    elseif name == "amgsa"
        return aspreconditioner(smoothed_aggregation(A)), true   # smoothed-aggregation AMG (cheaper setup)
    end
    error("unknown preconditioner: $name")
end

# ---------------- PCG with a TRUE-residual guard ----------------
# Krylov's cg stops on the PRECONDITIONED residual norm, relative to ITS OWN starting
# residual (handout §3.2 warning). The grader checks ‖b − A x‖₂ / ‖b‖₂. So we run,
# measure the true residual ourselves, and if it is not small enough we warm-start
# from x and ask for the reduction factor that is still missing (with a 2× margin).
function pcg(A, b, M, ldiv; rtol, itmax = 20_000)
    kw = M === nothing ? (;) : (; M = M, ldiv = ldiv)
    x, stats = cg(A, b; atol = 0.0, rtol = rtol, itmax = itmax, kw...)
    iters    = stats.niter
    rr       = relres(A, b, x)
    restarts = 0
    while rr > rtol && restarts < 4 && iters < itmax
        restarts += 1
        x, stats = cg(A, b, x; atol = 0.0, rtol = 0.5 * rtol / rr, itmax = itmax - iters, kw...)
        iters += stats.niter
        rr     = relres(A, b, x)
    end
    return x, iters, rr, restarts
end
