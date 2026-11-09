# -------------------------------------------------------
# Baseline for project 1
# LU with partial pivoting
# Respects column major memory format, but no blocking.
# ---------------------------------------------------------

# Find the maximum absolute value below the current pivot
function _findpivot(A, i)
    acol = @view A[i:end,i]
    max_val, max_idx = abs(acol[1]), 1
    for ii in eachindex(acol)
        candidate = abs(acol[ii])
        if candidate > max_val
            max_val, max_idx = candidate, ii
        end
    end
    return max_idx + i - 1
end

# Row swap. This is currently column-traversing, so not as efficient as possible
function _rowswap!(A::Matrix, r1, r2)
    r1 == r2 && return A
    for j in axes(A,2)
        A[r1,j], A[r2, j] = A[r2, j], A[r1, j]
    end
    return A
end

function lu(A::Matrix{T}) where T

    n, ncol = size(A)
    @assert n == ncol
    LU, p = copy(A), collect(1:n)

    @inbounds for i in 1:n
        # Find pivot, swap rows
        pivot_row = _findpivot(LU, i)
        # Row-swap
        _rowswap!(LU, i, pivot_row)
        p[i], p[pivot_row] = p[pivot_row], p[i]
        # Compute multipliers, stored in place below the diagonal
        pivot_val = LU[i,i]
        @simd for ii in (i+1):n
            LU[ii,i] /= pivot_val
        end
        # Row operations
        for j in (i+1):n
            LUij = LU[i,j]
            @simd for ii in (i+1):n
                LU[ii,j] -= LU[ii,i] * LUij
            end
        end
    end
    return (LU = LU, p = p)
end