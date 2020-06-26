#=
    To analyze the allocation during data generation
=#

import LinearAlgebra: mul!
import Distributions: Bernoulli
import StatsFuns: logistic

using TimerOutputs

function generate1!(y, X, β)
    @timeit "z = X * β" z = X * β 
    @timeit "p" p = logistic.(z)
    @timeit "y" y .= rand.(Bernoulli.(p))
    return y
end

function generate2!(y, X, β)
    @timeit "y .= X * β" y .= X * β  # reuse allocated "y"
    @timeit "p" p = logistic.(y)
    @timeit "y" y .= rand.(Bernoulli.(p))
    return y
end

function generate3!(y, X, β)
    @timeit "mul!(y, X, β)" mul!(y, X, β)  # reuse allocated "y"
    @timeit "p" p = logistic.(y)
    @timeit "y" y .= rand.(Bernoulli.(p))
end

# Test data
X = randn(1000, 50)
β = rand(size(X, 2))
# y = Vector(undef, size(X, 1))
y = Vector{eltype(X)}(undef, size(X, 1))

# do precompile - both the test functions as well as timers
reset_timer!()
@timeit "" generate1!(y, X, β);
@timeit "" generate2!(y, X, β);
@timeit "" generate3!(y, X, β)

# Actual test
reset_timer!()
for i = 1:100
@timeit "generate1!()" generate1!(y, X, β);
@timeit "generate2!()" generate2!(y, X, β);
@timeit "generate3!()" generate3!(y, X, β);
end
print_timer()
println()