# =====================================================================
# Question 2 — conjugate gradient under different preconditioners.
#
#   julia --project=. run_iterative.jl -d 3 --m 10,20,30,40 --rtol 1e-6
#       [--precond none,jacobi,ic,amg] [--itmax 20000] [--blas 1] [--out results/iterative.csv]
#
# For every (m, preconditioner, rtol): time the setup (building M) and the
# solve (all CG iterations + the true-residual check), record the iteration
# count and the achieved TRUE relative residual.
# BLAS is restricted to --blas threads (default 1) for the reason in the
# handout: multithreaded dot/axpy interleaved with single-threaded sparse
# products slows CG down. The previous value is restored at the end.
# =====================================================================
include(joinpath(@__DIR__, "common.jl"))

d        = getarg("-d", 2)
ms       = getints("--m", d == 2 ? [50, 100, 200, 400] : [10, 20, 30, 40])
rtols    = getfloats("--rtol", d == 2 ? [1e-8] : [1e-6])
preconds = getlist("--precond", ALL_PRECONDS)
itmax    = getarg("--itmax", 20_000)
blas     = getarg("--blas", 1)
out      = getstr("--out", "results/iterative.csv")

header = ["d", "m", "n", "nnzA", "precond", "rtol", "t_setup", "t_solve", "t_total",
          "iters", "restarts", "relres", "converged", "blas_threads", "status"]

function one_iter(d, m, pc, rtol; record = true)
    A, b = build(d, m)
    n = size(A, 1)
    bt = BLAS.get_num_threads()
    try
        t0 = time_ns();  M, ldiv = make_precond(pc, A);                      t_setup = (time_ns() - t0) / 1e9
        t0 = time_ns();  x, iters, rr, restarts = pcg(A, b, M, ldiv; rtol, itmax); t_solve = (time_ns() - t0) / 1e9
        conv = rr <= rtol
        record && logrow(out, header, [d, m, n, nnz(A), pc, rtol, t_setup, t_solve, t_setup + t_solve,
                                       iters, restarts, rr, conv, bt, "ok"])
        @printf("  %-7s rtol=%.0e  setup %.3fs  solve %.3fs  iters %6d  restarts %d  relres=%.1e %s\n",
                pc, rtol, t_setup, t_solve, iters, restarts, rr, conv ? "" : "NOT CONVERGED")
    catch err
        status = err isa OutOfMemoryError ? "OOM" : "error: " * string(typeof(err))
        record && logrow(out, header, [d, m, n, nnz(A), pc, rtol, "", "", "", "", "", "", "", bt, status])
        println("  ", rpad(pc, 8), status, "   ", first(sprint(showerror, err), 200))
    end
    flush(stdout)
end

blas_before = BLAS.get_num_threads()
BLAS.set_num_threads(blas)
println("Iterative experiments: d=$d  m=$ms  rtol=$rtols  precond=$preconds  BLAS threads=$blas  → $out")
try
    println("Warm-up (compilation) ...")
    for pc in preconds
        one_iter(d, d == 2 ? 8 : 4, pc, 1e-6; record = false)
    end
    for m in ms, rtol in rtols
        println("\n=== d=$d  m=$m  n=$(m^d)  rtol=$rtol ===")
        for pc in preconds
            one_iter(d, m, pc, rtol)
            GC.gc()
        end
    end
finally
    BLAS.set_num_threads(blas_before)      # leave the global setting as we found it
end
println("\nDone. Results appended to $out")
