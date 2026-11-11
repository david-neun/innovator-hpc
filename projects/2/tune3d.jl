# =====================================================================
# Project 2 — tuning solve3d at the official 3D size (m = 80, rtol = 1e-6).
# Each variant is a complete solve (setup + PCG + true-residual guard),
# timed like test.jl: fresh copies of A and b, best of 3 runs.
#     julia tune3d.jl            (on the grading node; see hpc_tune3d.sbatch)
# =====================================================================
using LinearAlgebra, SparseArrays, Printf
using Krylov, AlgebraicMultigrid
include(joinpath(@__DIR__, "problem.jl"))

const M3, RTOL = 80, 1e-6
relres(A, b, x) = norm(b - A * x) / norm(b)

# PCG with the true-residual guard; `tight` = factor applied to rtol inside Krylov
function pcg_guard(A, b, M, rtol; tight = 1.0, x0 = nothing, itmax = 20_000)
    x, stats = x0 === nothing ? cg(A, b;     M = M, ldiv = true, atol = 0.0, rtol = tight * rtol, itmax = itmax) :
                                cg(A, b, x0; M = M, ldiv = true, atol = 0.0, rtol = tight * rtol, itmax = itmax)
    iters = stats.niter; rr = relres(A, b, x); restarts = 0
    while rr > rtol && restarts < 5 && iters < itmax
        restarts += 1
        x, stats = cg(A, b, x; M = M, ldiv = true, atol = 0.0, rtol = 0.5 * rtol / rr, itmax = itmax - iters)
        iters += stats.niter; rr = relres(A, b, x)
    end
    return x, iters, restarts, rr
end

sa(A; kw...) = aspreconditioner(smoothed_aggregation(A; kw...))
rs(A; kw...) = aspreconditioner(ruge_stuben(A; kw...))

# name => (A, b) -> (x, iters, restarts, relres)
variants = [
 "sa_default"      => (A, b) -> pcg_guard(A, b, sa(A), RTOL),
 "sa_tight0.3"     => (A, b) -> pcg_guard(A, b, sa(A), RTOL; tight = 0.3),
 "sa_tight0.1"     => (A, b) -> pcg_guard(A, b, sa(A), RTOL; tight = 0.1),
 "sa_x0jacobi"     => (A, b) -> pcg_guard(A, b, sa(A), RTOL; tight = 0.3, x0 = b ./ diag(A)),
 "sa_mc50"         => (A, b) -> pcg_guard(A, b, sa(A; max_coarse = 50), RTOL; tight = 0.3),
 "sa_mc200"        => (A, b) -> pcg_guard(A, b, sa(A; max_coarse = 200), RTOL; tight = 0.3),
 "sa_ml6"          => (A, b) -> pcg_guard(A, b, sa(A; max_levels = 6), RTOL; tight = 0.3),
 "sa_gs2"          => (A, b) -> pcg_guard(A, b, sa(A; presmoother = GaussSeidel(iter = 2), postsmoother = GaussSeidel(iter = 2)), RTOL; tight = 0.3),
 "sa_fwdbwd"       => (A, b) -> pcg_guard(A, b, sa(A; presmoother = GaussSeidel(ForwardSweep(), 1), postsmoother = GaussSeidel(BackwardSweep(), 1)), RTOL; tight = 0.3),
 "sa_theta0.05"    => (A, b) -> pcg_guard(A, b, sa(A; strength = SymmetricStrength(0.05)), RTOL; tight = 0.3),
 "sa_theta0.25"    => (A, b) -> pcg_guard(A, b, sa(A; strength = SymmetricStrength(0.25)), RTOL; tight = 0.3),
 "rs_default"      => (A, b) -> pcg_guard(A, b, rs(A), RTOL),
 "rs_classical0.5" => (A, b) -> pcg_guard(A, b, rs(A; strength = Classical(0.5)), RTOL; tight = 0.3),
]

blas_before = BLAS.get_num_threads(); BLAS.set_num_threads(1)
A0, b0 = problem(M3; d = 3)
mkpath("results"); out = "results/tune3d_hpc.csv"
isfile(out) || open(out, "w") do io; println(io, "variant,t_best,t_runs,iters,restarts,relres,status"); end
println("tuning solve3d at d=3 m=$M3 rtol=$RTOL  (n = $(size(A0,1)))\n"); flush(stdout)
for (name, f) in variants
    try
        f(copy(A0), copy(b0))                    # warm-up (compilation of this variant)
        ts = Float64[]; local x, it, rst, rr
        for k in 1:3
            A, b = copy(A0), copy(b0)
            t0 = time_ns(); x, it, rst, rr = f(A, b); push!(ts, (time_ns() - t0) / 1e9)
        end
        ok = rr <= RTOL
        @printf("%-16s best %.3fs  runs %s  iters %3d  restarts %d  relres %.1e %s\n",
                name, minimum(ts), join([@sprintf("%.2f", t) for t in ts], "/"), it, rst, rr, ok ? "" : "FAILS TOLERANCE")
        open(out, "a") do io; println(io, join([name, minimum(ts), join(ts, ";"), it, rst, rr, ok ? "ok" : "fails"], ",")); end
    catch err
        msg = first(sprint(showerror, err), 160)
        @printf("%-16s ERROR %s\n", name, msg)
        open(out, "a") do io; println(io, join([name, "", "", "", "", "", "error: " * replace(msg, "," => ";")], ",")); end
    end
    flush(stdout); GC.gc()
end
BLAS.set_num_threads(blas_before)
println("\ndone → $out")
