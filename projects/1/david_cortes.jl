# ---------------------------------------------------------------
# LU decomposition with partial pivoting, packed storage.
# Returns (LU = LU, p = p) such that  L*U == A[p, :],
#   - below the diagonal of LU: the multipliers (this is L)
#   - on and above the diagonal of LU: the entries of U
#   - the unit diagonal of L is implicit (not stored)
# ---------------------------------------------------------------
function lu(A::Matrix{T}) where T<:Real
    n = size(A, 1)

    # ----- Part 1: setup -------------------------------------------
    # Work on a private Float64 copy so that:
    #   (a) A is never modified (test.jl checks this), and
    #   (b) an Int input can't cause integer division.
    # Scalar loops, j outer / i inner: down a column, matching
    # Julia's column-major storage.
    LU = Matrix{Float64}(undef, n, n)
    for j in 1:n
        for i in 1:n
            LU[i, j] = A[i, j]
        end
    end

    # p[i] = index of the ORIGINAL row that currently sits in row i.
    # Starts as the identity permutation (1, 2, ..., n).
    p = collect(1:n)

    # ----- Main loop: one column of elimination per iteration -------
    for k in 1:(n-1)

        # ----- Part 2: pivot search ---------------------------------
        # Find the row (k..n) whose entry in column k has the largest
        # absolute value. The strict ">" keeps the EARLIEST row on ties,
        # which is the tie-breaking rule used in the worked examples.
        piv  = k
        best = abs(LU[k, k])
        for i in (k+1):n
            v = abs(LU[i, k])
            if v > best
                best = v
                piv  = i
            end
        end

        # ----- Part 3: row swap -------------------------------------
        # Swap the ENTIRE rows k and piv (columns 1:n, not k:n), so the
        # multipliers already stored in columns 1:k-1 travel with their
        # rows. Swapping only k:n is the bug shown in Example 5.
        if piv != k
            for j in 1:n
                tmp         = LU[k, j]
                LU[k, j]    = LU[piv, j]
                LU[piv, j]  = tmp
            end
            # Record the same swap in the permutation vector.
            tmp_p  = p[k]
            p[k]   = p[piv]
            p[piv] = tmp_p
        end

        # ----- Part 4a: multipliers (this becomes L) -----------------
        # m_ik = LU[i,k] / pivot, stored in the position it zeroes out.
        # After this loop, LU[i,k] holds the multiplier m_ik for i > k.
        for i in (k+1):n
            LU[i, k] = LU[i, k] / LU[k, k]
        end

        # ----- Part 4b: update the trailing block (this becomes U) ---
        # LU[i,j] <- LU[i,j] - m_ik * LU[k,j]   for i, j > k.
        # j outer, i inner: the inner loop walks down column j, which
        # is contiguous in memory. LU[k,j] doesn't depend on i, so it is
        # read once per column (ukj) instead of once per entry.
        # Step 5 later: put @inbounds @simd on the inner loop only,
        # after the correctness tests pass.
        for j in (k+1):n
            ukj = LU[k, j]
            for i in (k+1):n
                LU[i, j] -= LU[i, k] * ukj
            end
        end
    end

    return (LU = LU, p = p)
end