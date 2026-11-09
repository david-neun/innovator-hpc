using BenchmarkTools

module Submission
    include("david_cortes.jl")   # same wrapper test.jl uses
end

A = randn(500, 500)

@btime Submission.lu($A)                # one line: minimum time + allocations
display(@benchmark Submission.lu($A))   # full report, like your screenshot
t = @belapsed Submission.lu($A)         # minimum time in seconds, as a number