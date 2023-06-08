using Plots
using Distributions
using LinearAlgebra

##
μ₁ = [1.0, 1.0]
Σ₁ = [2.0 0.65; 0.65 1.0]

μ₂ = -1.0 .* μ₁
Σ₂ = [2.0 -0.65; -0.65 1.0]
Σ₂ = Σ₂' * Σ₂

##
D₁ = MultivariateNormal(μ₁, Σ₁)
D₂ = MultivariateNormal(μ₂, Σ₂)

##
N = 100
s = hcat([rand(D₁, N) rand(D₂, N)])

##
plot()
scatter!(s[1, :], s[2, :], label=nothing)

##
r = s' * s

plot()
heatmap!(r)

##