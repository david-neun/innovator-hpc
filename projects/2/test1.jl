# --------------------------------------
# Project 2 Testing Harness
# --------------------------------------

# Builds the system A x = b from problem.jl, and then
# 1) Confirms your solve meets the requested relative residual
# 2) Times your solve using @btime
# Run from the command line for a given dimension, grid size, and tolerance with:
#   julia test.jl -d 3 -m 40 --rtol 1e-6
# ...to run the test on a 40 x 40 x 40 grid with rtol = 1e-6.

# REPLACE THE ARGUMENT BELOW WITH THE RELATIVE PATH TO YOUR FUNCTION SCRIPT !!!
# Your file is loaded into its own module, so it must load every package it uses.
FILE_PATH = "baseline.jl"
module Submission
    include(Main.FILE_PATH)
end
@info "Testing $FILE_PATH"
# ---------------------------------------
# Imports and functions
# ---------------------------------------
using BenchmarkTools
using LinearAlgebra, SparseArrays
include(joinpath(@__DIR__, "problem.jl"))

# Optional command line arguments. Example : julia test.jl -d 3 -m 40 --rtol 1e-6
# Defaults to d = 2, m = 200, rtol = 1e-8
function getarg(flag, default)
    i = findfirst(==(flag), ARGS)
    return i === nothing ? default : parse(typeof(default), ARGS[i+1])
end
d    = getarg("-d", 2)
m    = getarg("-m", 200)
rtol = getarg("--rtol", 1e-8)

@info "Running with d=$d, m=$m (n=$(m^d)), rtol=$rtol"
@info "Julia threads: $(Threads.nthreads()), BLAS threads: $(BLAS.get_num_threads())"

# --------------------------------------
# Generate the system
# --------------------------------------

A, b = problem(m; d)
A_reference, b_reference = copy(A), copy(b) # Kept so we can detect a solve that clobbers A or b

# ------------------------------------
# Test your function
# ------------------------------------

x = Submission.solve(A, b; rtol)

# Your function must not modify its arguments; the benchmark below calls it
# many times on the same system.
A == A_reference || error("Your `solve` modified A. It must leave A unchanged.")
b == b_reference || error("Your `solve` modified b. It must leave b unchanged.")

# Check the shape of what came back before trusting it
x isa AbstractVector{<:Real} || error("Your `solve` must return a vector, got $(typeof(x)).")
length(x) == length(b) || error("`x` must have length $(length(b)), got $(length(x)).")

# Check the true relative residual of the original system
relres = norm(b - A * x) / norm(b)
relres <= rtol || error("Your solution does not meet the tolerance: relative residual $relres > rtol = $rtol.")
@info "Your solution meets the tolerance (relative residual $(round(relres, sigdigits = 2)))."

# ------------------------------------
# Benchmark your function
# ------------------------------------

# Each sample gets fresh copies of A and b, built outside the timed region,
# so nothing from one call can be reused by the next.
@info "Benchmarking..."
@btime Submission.solve(Ac, bc; rtol = $rtol) setup = (Ac = copy($A); bc = copy($b)) evals = 1
