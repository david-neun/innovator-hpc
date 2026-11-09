# ---------------------------------------------------------------
# Project 2 — package setup.  Run ONCE from inside projects/2:
#     julia setup_env.jl              # installs into this folder's environment
#     julia setup_env.jl --global     # installs into Julia's default environment
#                                     # (needed on the cluster: test.jl is run
#                                     #  there WITHOUT --project, like the grader does)
# ---------------------------------------------------------------
import Pkg

if !("--global" in ARGS)
    Pkg.activate(@__DIR__)          # creates Project.toml / Manifest.toml here
end

pkgs = [
    "AMD",                        # approximate minimum degree ordering
    "SymRCM",                     # reverse Cuthill–McKee ordering
    "Metis",                      # nested dissection ordering
    "Krylov",                     # conjugate gradient
    "AlgebraicMultigrid",         # AMG preconditioner
    "LimitedLDLFactorizations",   # incomplete Cholesky (limited-memory LDLᵀ)
    "BenchmarkTools",             # used by test.jl
]
Pkg.add(pkgs)
Pkg.precompile()
println("\nPackages ready: ", join(pkgs, ", "))
