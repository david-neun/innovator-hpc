module Submission
    include("david_cortes.jl")
end

# Each case: (input matrix, expected packed LU, expected p)
cases = [
    ([0.0 1 2; 2 4 3; 4 6 1], [4 6 1; 0.5 1 2.5; 0 1 -0.5],      [3, 2, 1]),
    ([1.0 1 1; 4 2 3; 1 1 2], [4 2 3; 0.25 0.5 0.25; 0.25 1 1],  [2, 1, 3]),
    ([1.0 1 1; 2 2 1; 2 1 2], [2 2 1; 1 -1 1; 0.5 0 0.5],        [2, 3, 1]),
]

for (idx, (A, LU_expected, p_expected)) in enumerate(cases)
    r = Submission.lu(A)
    println("Case $idx")
    println("  LU matches: ", isapprox(r.LU, LU_expected))
    println("  p matches:  ", r.p == p_expected)
    println("  your p = ", r.p)
    println("  your LU:")
    display(r.LU)
    println()
end