# Find the maximal pivot for the ith elimination step
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

# Swap two rows in place
function _rowswap!(A, r1, r2)
    r1 == r2 && return A
    for j in axes(A,2)
        A[r1,j], A[r2, j] = A[r2, j], A[r1, j]
    end
    return A
end


# Compute the LU factorization of the left rectangular panel, between columns j1 and j2, rows below j1
function _panellu!(A, p, j1, j2)
    Abelow = @view A[j1:end,:]
    LU = @view Abelow[:,j1:j2]
    q = @view p[j1:end]
    n,m = size(LU)
    @inbounds for i in 1:m
        pivot_row = _findpivot(LU, i)
        _rowswap!(Abelow, i, pivot_row)
        q[i], q[pivot_row] = q[pivot_row], q[i]
        pivot_val = LU[i,i]
        @simd for ii in (i+1):n
            LU[ii,i] /= pivot_val
        end
        for j in (i+1):m
            LUij = LU[i,j]
            @simd for ii in (i+1):n
                LU[ii,j] -= LU[ii,i] * LUij
            end
        end
    end
end

# Scalar, non-blocked C -= A * B
function _mulsub!(C, A, B)
    @inbounds for j in axes(A,2)
        for k in axes(C,2)
            b_jk = B[j,k]
            @simd for i in axes(A,1)
                C[i,k] -= b_jk * A[i,j]
            end
        end
    end
    return C
end

# Blocked C -= A*B
function _blockmulsub!(C, A, B; bs::Int = 64)
    n,m = size(A)
    m_,p = size(B)
    n_, p_ = size(C)
    @assert m == m_ # Make sure inner dims align
    @assert (n_,p_) == (n, p) # Make sure outer dims align
    @inbounds for j in 1:bs:m
        j_range = j:min(j+bs-1, m)
        for k in 1:bs:p
            k_range = k:min(k+bs-1,p)
            B_jk = @view B[j_range,k_range]
            for i in 1:bs:n
                i_range = i:min(i+bs-1,n)
                C_ik = @view C[i_range, k_range]
                A_ij = @view A[i_range, j_range]
                _mulsub!(C_ik, A_ij, B_jk)
            end
        end
    end
    return C
end 

# Solve Lz = x, storing the result in x  ( x <- z )
function _forwardsub!(x::AbstractVector, L::AbstractMatrix)
    n = length(x)
    @inbounds for j in axes(L,2)
        xj = x[j]
        @simd for i in (j+1):n
            x[i] -= xj*L[i,j]
        end
    end
    return x
end

# Same as above, for for a matrix of right hand sides
function _forwardsub!(X::AbstractMatrix, L::AbstractMatrix)
    @inbounds for k in axes(X,2)
        x = @view X[:,k]
        _forwardsub!(x, L)
    end
    return X
end

# Blocked LU. Block size of `bs_lu` for the LU partitioning,
# block size of `bs_mm` for the matrix-matrix multiplication
function lu(A; bs_lu::Int = 64, bs_mm::Int = 256)
    n, ncol = size(A)
    LU = copy(A)
    p = collect(1:n)
    @assert n == ncol
    for j in 1:bs_lu:n
        jend = min(j+bs_lu-1, n)
        idx1 = j:jend
        idx2 = (jend+1):n
        # Panel LU factorization
        _panellu!(LU, p, j, jend)
        # Right-looking forward solve
        L11 = @view LU[idx1,idx1]
        A12 = @view LU[idx1,idx2]
        _forwardsub!(A12, L11)
        # Trailing update
        A22 = @view LU[idx2,idx2]
        A21 = @view LU[idx2,idx1]
        _blockmulsub!(A22, A21, A12; bs = bs_mm)
    end
    return (LU = LU, p = p)
end





