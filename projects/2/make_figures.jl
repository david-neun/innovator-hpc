# =====================================================================
# Project 2 — figures for the report, from the CSVs written by
# run_direct.jl / run_iterative.jl on the cluster.
#     julia make_figures.jl            (see hpc_plots.sbatch)
# Output: plot/fig1_orderings.png, plot/fig2_preconditioners.png,
#         plot/fig3_crossover.png, plot/slopes.txt
# Needs Plots.jl:  julia -e 'using Pkg; Pkg.add("Plots")'
# =====================================================================
ENV["GKSwstype"] = "100"            # headless plotting on the compute node
using Plots, Printf

# ---------------- tiny CSV reader (no extra packages) ----------------
function readcsv(path)
    lines = filter(!isempty, readlines(path))
    header = String.(split(lines[1], ","))
    rows = Dict{String,Any}[]
    for l in lines[2:end]
        f = String.(split(l, ","))
        length(f) == length(header) || continue
        d = Dict{String,Any}()
        for (h, v) in zip(header, f)
            x = tryparse(Float64, v)
            d[h] = x === nothing ? v : x
        end
        push!(rows, d)
    end
    return rows
end
csvfiles(pattern) = filter(f -> occursin(pattern, f) && endswith(f, ".csv"), readdir("results"; join = true))

# keep the fastest run when a (key) was measured more than once (like @btime does)
function dedup(rows, keyfields)
    best = Dict{Any,Dict{String,Any}}()
    for r in rows
        r["status"] == "ok" || continue
        k = Tuple(r[f] for f in keyfields)
        if !haskey(best, k) || r["t_total"] < best[k]["t_total"]
            best[k] = r
        end
    end
    return collect(values(best))
end

D = dedup(vcat([readcsv(f) for f in csvfiles("direct")]...), ["d", "m", "ordering"])
Iall = vcat([readcsv(f) for f in csvfiles(r"^results/(iterative|tune_hpc|tune_blas_hpc)")]...)
Iall = filter(r -> r["blas_threads"] == 1.0, Iall)
I = dedup(Iall, ["d", "m", "precond", "rtol"])

sel(rows; kw...) = sort(filter(r -> all(r[string(k)] == v for (k, v) in pairs(kw)), rows), by = r -> r["n"])
col(rows, f) = [r[f] for r in rows]
slope(x, y) = length(x) >= 2 ? (X = log.(x); Y = log.(y); (sum((X .- sum(X)/length(X)) .* (Y .- sum(Y)/length(Y))) / sum((X .- sum(X)/length(X)).^2))) : NaN

mkpath("plot")
default(titlefontsize = 9, guidefontsize = 8, tickfontsize = 7, legendfontsize = 6,
        markersize = 3, linewidth = 1, grid = true, minorgrid = true, dpi = 150)

ORD   = ["natural", "rcm", "amd", "metis", "default"]
ORDL  = Dict("natural" => "natural", "rcm" => "RCM", "amd" => "AMD", "metis" => "METIS (ND)", "default" => "cholesky(A) default")
ORDM  = Dict("natural" => :circle, "rcm" => :rect, "amd" => :utriangle, "metis" => :diamond, "default" => :xcross)
PC    = ["none", "jacobi", "ic", "amg", "amgsa"]
PCL   = Dict("none" => "CG, none", "jacobi" => "Jacobi", "ic" => "IC (lldl, mem 0)", "amg" => "AMG (Ruge–Stüben)", "amgsa" => "AMG (smoothed aggr.)")
PCM   = Dict("none" => :circle, "jacobi" => :rect, "ic" => :utriangle, "amg" => :diamond, "amgsa" => :pentagon)
official_rtol = Dict(2 => 1e-8, 3 => 1e-6)
report = String[]

# ---------------- Figure 1: orderings ----------------
panels = []
for d in (2, 3), what in ("fill_ratio", "t_total")
    p = plot(xscale = :log10, yscale = :log10, xlabel = "n = m^d",
             title = what == "fill_ratio" ? "$(d)D: fill ratio nnz(L)/nnz(A)" : "$(d)D: total time, order + factor + solve [s]",
             legend = (d == 2 && what == "fill_ratio") ? :topleft : false)
    for o in ORD
        s = sel(D; d = Float64(d), ordering = o)
        isempty(s) && continue
        plot!(p, col(s, "n"), col(s, what), label = ORDL[o], marker = ORDM[o], linestyle = o == "default" ? :dash : :solid)
        what == "fill_ratio" && push!(report, @sprintf("Q1 d=%d %-8s nnzL ~ n^%.2f   t_total ~ n^%.2f   largest n=%d: fill=%.1f t_total=%.3fs",
                                      d, o, slope(col(s, "n"), col(s, "nnzL")), slope(col(s, "n"), col(s, "t_total")), s[end]["n"], s[end]["fill_ratio"], s[end]["t_total"]))
    end
    push!(panels, p)
end
savefig(plot(panels[1], panels[3], panels[2], panels[4], layout = (2, 2), size = (1100, 800)), "plot/fig1_orderings.png")

# ---------------- Figure 2: preconditioners ----------------
panels = []
for d in (2, 3), what in ("iters", "t_total")
    rt = official_rtol[d]
    p = plot(xscale = :log10, yscale = :log10, xlabel = "n = m^d",
             title = what == "iters" ? "$(d)D: CG iterations to true rtol = $rt" : "$(d)D: total time, setup + iterations [s]",
             legend = (d == 2 && what == "iters") ? :bottomright : false)
    for pc in PC
        s = sel(I; d = Float64(d), precond = pc, rtol = rt)
        isempty(s) && continue
        plot!(p, col(s, "n"), col(s, what), label = PCL[pc], marker = PCM[pc])
        nc = filter(r -> r["converged"] == "false", s)
        isempty(nc) || scatter!(p, col(nc, "n"), col(nc, what), marker = :xcross, markersize = 7, color = :red, label = "")
        if what == "iters"
            c = filter(r -> r["converged"] == "true", s)
            push!(report, @sprintf("Q2 d=%d %-7s iters ~ n^%.2f (= m^%.2f)   t_total ~ n^%.2f   largest n=%d: iters=%d setup=%.3fs solve=%.3fs",
                  d, pc, slope(col(c, "n"), col(c, "iters")), d * slope(col(c, "n"), col(c, "iters")), slope(col(c, "n"), col(c, "t_total")),
                  s[end]["n"], s[end]["iters"], s[end]["t_setup"], s[end]["t_solve"]))
        end
    end
    what == "iters" && d == 3 && annotate!(p, 1.5e3, maximum(col(sel(I; d = 3.0, rtol = rt), "iters")) * 0.9, text("red x = not converged in 20 000 it.", 6, :left))
    push!(panels, p)
end
savefig(plot(panels[1], panels[3], panels[2], panels[4], layout = (2, 2), size = (1100, 800)), "plot/fig2_preconditioners.png")

# ---------------- Figure 3: crossover ----------------
p3 = []
for d in (2, 3)
    rt = official_rtol[d]
    p = plot(xscale = :log10, yscale = :log10, xlabel = "n = m^d", title = "$(d)D, rtol $rt [s]", legend = :topleft)
    sa = sel(D; d = Float64(d), ordering = "amd");   plot!(p, col(sa, "n"), col(sa, "t_total"), label = "Cholesky, AMD", marker = :xcross, color = :black, linestyle = :dash)
    sm = sel(D; d = Float64(d), ordering = "metis"); plot!(p, col(sm, "n"), col(sm, "t_total"), label = "Cholesky, METIS", marker = :diamond, color = :gray, linestyle = :dash)
    for pc in ("jacobi", "ic", "amg", "amgsa")
        s = filter(r -> r["converged"] == "true", sel(I; d = Float64(d), precond = pc, rtol = rt))
        isempty(s) && continue
        plot!(p, col(s, "n"), col(s, "t_total"), label = "PCG " * PCL[pc], marker = PCM[pc])
        ratios = join([@sprintf("n=%d: %.2f", r["n"], r["t_total"] / first(filter(q -> q["n"] == r["n"], sa))["t_total"]) for r in s if any(q -> q["n"] == r["n"], sa)], ", ")
        push!(report, "Q3 d=$d $pc / Cholesky-AMD time ratio — " * ratios)
    end
    push!(p3, p)
end
pr = plot(xscale = :log10, yscale = :log10, xlabel = "rtol", title = "PCG time / Cholesky time vs rtol", legend = :topright)
for (d, mbig, ls) in ((2, 500.0, :solid), (3, 80.0, :dash))
    tdir = first(sel(D; d = Float64(d), ordering = "amd", m = mbig))["t_total"]
    for pc in ("jacobi", "ic", "amg")
        s = sort(filter(r -> r["d"] == d && r["m"] == mbig && r["precond"] == pc, I), by = r -> r["rtol"])
        isempty(s) && continue
        plot!(pr, col(s, "rtol"), col(s, "t_total") ./ tdir, label = "$(d)D m=$(Int(mbig)) $(PCL[pc])", marker = PCM[pc], linestyle = ls)
        push!(report, "rtol d=$d m=$(Int(mbig)) $pc: " * join([@sprintf("%.0e -> %d it, %.3fs", r["rtol"], r["iters"], r["t_total"]) for r in s], "; "))
    end
end
hline!(pr, [1.0], color = :black, label = "= Cholesky (AMD)")
savefig(plot(p3[1], p3[2], pr, layout = (1, 3), size = (1250, 420)), "plot/fig3_crossover.png")

open("plot/slopes.txt", "w") do io; foreach(l -> println(io, l), report); end
println(join(report, "\n")); println("\nfigures written to plot/")
