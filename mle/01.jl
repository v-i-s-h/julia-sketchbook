#=
Source: https://github.com/johnmyleswhite/julia_tutorials/blob/master/Statistics%20in%20Julia%20-%20Maximum%20Likelihood%20Estimation.ipynb
=#

import LinearAlgebra: diag, dot, mul!
import Statistics: cov, mean, var

import Distributions: Bernoulli, Normal, cquantile
import ForwardDiff: hessian
import Optim: LBFGS, minimizer, optimize
import StatsFuns: logistic, log1pexp, logit

## Data Generation
function generate!(y, X, β)
    mul!(y, X, β)
    p = logistic.(y)
    y .= rand.(Bernoulli.(p))
end

function simulate_data(n, d)
    X = hcat(ones(n), rand(Normal(0, 1), n, d))
    β = rand(Normal(0, 1), d + 1)
    y = Array{Float64}(undef, n)
    generate!(y, X, β)
    y, X, β
end

## Log Likelihood function
function log_likelihood(X, y, β)
    ll = 0.0
    @inbounds for i in eachindex(y)
        zᵢ = @view(X[i,:])' * β
        c = -log1pexp(-zᵢ)
        ll += y[i] * c + (1 - y[i]) * (-zᵢ + c)
    end
    ll
end

n = 500
d = 2
y, X, β = simulate_data(n, d)

# create a closure for optimization
make_closures(X, y) = β -> -log_likelihood(X, y, β)
nll = make_closures(X, y)

## Minimizing the NLL
β₀ = zeros(d + 1)    # Some starting point

# # compute solution
r1 = optimize(nll, β₀, LBFGS(), autodiff=:forward)
# print(r1)

## Minimization with robust initialization
function initialize!(β₀, X, y, ϵ = 0.1)
    β₀[1] = log(mean(y))
    logit_y = [ifelse(yᵢ == 1.0, logit(1 - ϵ), logit(ϵ)) for yᵢ in y]
    for j in 2:length(β₀)
        β₀[j] = cov(logit_y, @view(X[:, j])) / var(@view(X[:, j]))
    end
    β₀
end

initialize!(β₀, X, y)
r2 = optimize(nll, β₀, LBFGS(), autodiff=:forward)
# print(r2)

## Test estimates
function compute_ses(nll, β̂)
    H = hessian(nll, β̂)
    ses = .√(diag(inv(H)))
    ses
end

function compute_cis(nll, β̂, α)
    ses = compute_ses(nll, β̂)
    τ = cquantile(Normal(0, 1), α)
    l = β̂ - τ * ses
    u = β̂ + τ * ses
    l, u
end

check_cis(β, l, u) = all(l .<= β .<= u)

α = 0.001

β₁ = minimizer(r1)
β₂ = minimizer(r2)

# function to nicely format vectors
stringify(vector) = "[" * join(map(v -> @sprintf("%6.5f", v), vector), ", ") * "]"

println("β  = ", stringify(β), "   ll = ", log_likelihood(X, y, β))

println("β₁ = ", stringify(β₁), "   ll = ", log_likelihood(X, y, β₁))
l, u = compute_cis(nll, β₁, 0.10)
println("    α = 0.1000", "\n",
        "        l = ", stringify(l), "\n", 
        "        u = ", stringify(u))
l, u = compute_cis(nll, β₁, 0.01)
println("    α = 0.0100", "\n",
        "        l = ", stringify(l), "\n", 
        "        u = ", stringify(u))
l, u = compute_cis(nll, β₁, 0.001)
println("    α = 0.0010", "\n",
        "        l = ", stringify(l), "\n", 
        "        u = ", stringify(u))
l, u = compute_cis(nll, β₁, 0.0001)
println("    α = 0.0001", "\n",
        "        l = ", stringify(l), "\n", 
        "        u = ", stringify(u))
        
println("β₂ = ", stringify(β₂), "   ll = ", log_likelihood(X, y, β₂))
l, u = compute_cis(nll, β₂, 0.10)
println("    α = 0.1000", "\n",
        "        l = ", stringify(l), "\n", 
        "        u = ", stringify(u))
l, u = compute_cis(nll, β₂, 0.01)
println("    α = 0.0100", "\n",
        "        l = ", stringify(l), "\n", 
        "        u = ", stringify(u))
l, u = compute_cis(nll, β₂, 0.001)
println("    α = 0.0010", "\n",
        "        l = ", stringify(l), "\n", 
        "        u = ", stringify(u))
l, u = compute_cis(nll, β₂, 0.0001)
println("    α = 0.0001", "\n",
        "        l = ", stringify(l), "\n", 
        "        u = ", stringify(u))
