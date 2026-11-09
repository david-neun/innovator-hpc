# =====================================================================
# Question 1 — sparse Cholesky under different orderings.
#
#   julia --project=. run_direct.jl -d 2 --m 50,100,200,400,800
#       [--orderings default,natural,rcm,amd,metis] [--maxgb 4] [--out results/direct.csv]
#
# For every (m, ordering): time the ordering, the factorization and the
# solve; record nnz(L), the fill ratio nnz(L)/nnz(A) and the true residual.
# natural/rcm are skipped when the band bound predicts more than --maxgb GB.
# One CSV row per run; the file is appended to, so runs can be split.
# =====================================================================
include(joinpath(@__DIR__, "common.jl"))

d         = getarg("-d", 2)
ms        = getints("--m", d == 2 ? [50, 100, 200, 400] : [10, 20, 30, 40])
orderings = getlist("--orderings", ALL_ORDERINGS)
maxgb     = getarg("--maxgb", 4.0)
out       = getstr("--out", "results/direct.csv")

header = ["d", "m", "n", "nnzA", "ordering", "t_order", "t_factor", "t_solve", "t_total",
          "nnzL", "fill_ratio", "relres", "default_perm_equals_amd", "blas_threads", "status"]

function one_direct(d, m, ordering; record = true)
    A, b = build(d, m)
    n = size(A, 1)
    bt = BLAS.get_num_threads()

    if ordering in ("natural", "rcm") && band_bytes(d, m) > maxgb * 1e9
        msg = @sprintf("skipped: band bound %.1f GB > %.1f GB", band_bytes(d, m) / 1e9, maxgb)
        record && logrow(out, header, [d, m, n, nnz(A), ordering, "", "", "", "", "", "", "", "", bt, msg])
        println("  ", rpad(ordering, 8), msg)
        return
    end

    try
        t0 = time_ns();  p = ordering_perm(ordering, A);                      t_order  = (time_ns() - t0) / 1e9
        t0 = time_ns();  F = (p === nothing ? cholesky(A) : cholesky(A; perm = p)); t_factor = (time_ns() - t0) / 1e9
        t0 = time_ns();  x = F \ b;                                            t_solve  = (time_ns() - t0) / 1e9
        nnzL = factor_nnz(F)
        rr   = relres(A, b, x)
        same = ordering == "default" ? string(F.p == ordering_perm("amd", A)) : ""
        record && logrow(out, header, [d, m, n, nnz(A), ordering, t_order, t_factor, t_solve,
                                       t_order + t_factor + t_solve, nnzL, nnzL / nnz(A), rr, same, bt, "ok"])
        @printf("  %-8s order %.3fs  factor %.3fs  solve %.3fs  nnz(L)=%.3g  fill=%.1f  relres=%.1e\n",
                ordering, t_order, t_factor, t_solve, nnzL, nnzL / nnz(A), rr)
    catch err
        status = err isa OutOfMemoryError ? "OOM" : "error: " * string(typeof(err))
        record && logrow(out, header, [d, m, n, nnz(A), ordering, "", "", "", "", "", "", "", "", bt, status])
        println("  ", rpad(ordering, 8), status)
    end
    flush(stdout)
end

println("Direct experiments: d=$d  m=$ms  orderings=$orderings  → $out")
println("Warm-up (compilation) ...")
for o in orderings
    one_direct(d, d == 2 ? 8 : 4, o; record = false)
end
for m in ms
    println("\n=== d=$d  m=$m  n=$(m^d) ===")
    for o in orderings
        one_direct(d, m, o)
        GC.gc()                     # give the memory of the last factor back before the next one
    end
end
println("\nDone. Results appended to $out")
