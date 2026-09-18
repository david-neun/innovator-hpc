# --------------------------------------
# Project 1 Testing Harness
# --------------------------------------

# Simulates a random n x n L and U, forms A, and then
# 1) Confirms your packed factorization recovers the correct L, U
# 2) Times your factorization using @btime
# Run from the command line for a given `n` with :
#   julia test.jl -n 500
# ...to run the test with n = 500.

# REPLACE THE ARGUMENT BELOW WITH THE RELATIVE PATH TO YOUR FUNCTION SCRIPT !!!
# Your file is loaded into its own module, so it must stand on its own: it
# gets Base and nothing else, which is what Requirement 2 asks for anyway.
FILE_PATH = "mylu.jl"
module Submission
    include(Main.FILE_PATH) 
end
@info "Testing $FILE_PATH"
# ---------------------------------------
# Imports and functions
# ---------------------------------------
using Random # Required to set the random seed for reproducibility.
using BenchmarkTools
using LinearAlgebra # Used only to check your result. Your submission is loaded
                    # into its own module and does NOT get access to this.

# A function to randomly generate a unit lower triangular L and an upper
# triangular U, whose product is the test matrix.
function randomLU(n, T)
    L = zeros(T, n, n)
    U = zeros(T, n, n)
    for i in 1:n
        L[i,i] = one(T)
        U[i,i] = randn()
        for j in (i+1):n
            L[j, i] = randn()
            U[i, j] = randn()
        end
    end
    return L, U
end

# Optional command line size. Example : julia test.jl -n 1_000
# Defaults to n = 500
n = (i = findfirst(==("-n"), ARGS)) === nothing ? 500 : parse(Int, ARGS[i+1])
T = Float64

@info "Running with n=$n"

# --------------------------------------
# Generate true L, U, and A
# --------------------------------------

Random.seed!(123) # Set the random seed for reproducibility
L, U = randomLU(n, T)
A = L*U
A_reference = copy(A) # Kept so we can detect a factorization that clobbers A

# ------------------------------------
# Test your function
# ------------------------------------

# Compute the LU decompostion
mylu = Submission.lu(A)

# Your function must not modify its argument; the benchmark below calls it
# many times on the same matrix.
A == A_reference || error("Your `lu` modified its input matrix. It must leave A unchanged.")

# Check the shape of what came back before trusting it
hasproperty(mylu, :LU) || error("Your `lu` must return a named tuple with a field `LU`.")
hasproperty(mylu, :p)  || error("Your `lu` must return a named tuple with a field `p`.")
size(mylu.LU) == (n, n) || error("`LU` must be $n x $n, got $(size(mylu.LU)).")
sort(mylu.p) == collect(1:n) || error("`p` must be a permutation of 1:$n.")

# Check if the decompostion is correct, up to permutation and floating point error.
# L and U are the two triangles of the packed result: multipliers below the
# diagonal, U on and above it, L's unit diagonal implicit. These wrappers read
# them in place -- nothing is copied.
myL = UnitLowerTriangular(mylu.LU)
myU = UpperTriangular(mylu.LU)
A_test = myL * myU
correct = isapprox(A_test, A[mylu.p,:])
if correct
    @info "Your factorization is correct."
else
    error("Your factorization is incorrect.")
end

# Benchmark
@info "Benchmarking..."
@btime Submission.lu($A);
