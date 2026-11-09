# --------------------------------------
# Project 2 problem generator
# --------------------------------------

# Builds the sparse SPD system A x = b for steady-state diffusion
#   -div( k(z) grad u(z) ) = 1   on the unit square (d = 2) or cube (d = 3),
# with u = 0 on the boundary, on a grid with m unknowns per side (n = m^d).
# Usage:
#   include("problem.jl")
#   A, b = problem(200; d = 2)
# Unknowns are numbered in the natural (column-major) order.

using SparseArrays

# Permeability. Varies smoothly between 10^-2 and 10^2: four orders of magnitude.
permeability(z...) = 10.0^(2 * prod(sinpi(2 * (i + 1) * zi) for (i, zi) in enumerate(z)))

function problem(m::Integer; d::Integer = 2)
    d in (2, 3) || throw(ArgumentError("d must be 2 or 3, got $d"))
    m >= 1 || throw(ArgumentError("m must be positive, got $m"))
    h = 1 / (m + 1)
    n = m^d
    idx = LinearIndices(ntuple(_ -> m, d))
    I, J, V = Int[], Int[], Float64[]
    diagonal = zeros(n)
    for c in CartesianIndices(idx)
        i = idx[c]
        ki = permeability((Tuple(c) .* h)...)
        for dir in 1:d, s in (-1, 1)
            cn = c + CartesianIndex(ntuple(t -> t == dir ? s : 0, d))
            kn = permeability((Tuple(cn) .* h)...)
            kface = 2ki * kn / (ki + kn)    # harmonic average of k across the face
            diagonal[i] += kface
            # Neighbors outside the grid are boundary points, where u = 0
            if checkbounds(Bool, idx, cn)
                push!(I, i); push!(J, idx[cn]); push!(V, -kface)
            end
        end
    end
    A = sparse(vcat(I, 1:n), vcat(J, 1:n), vcat(V, diagonal), n, n) ./ h^2
    b = ones(n)
    return A, b
end
