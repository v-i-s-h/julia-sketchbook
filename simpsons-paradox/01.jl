# Script to vaisualize Simpson's paradox

using LinearAlgebra
using Distributions
using Plots, StatsPlots
using Printf
using DataFrames
using GLM

# function to generate centeroids in  line ax + b
function centeroids(K; a = 1.0, b = 0.0, c1 = -15.0, c2 = +15.0)
    x = rand(Uniform(c1, c2), K)
    y = a .* x .+ b

    return [ [_x, _y] for (_x, _y) ∈ zip(x, y) ]
end

function covmatrices(K; β = 1.0, λ = 0.0)
    s = [rand(2, 2) for i ∈ 1:K]
    Σ = map(_s -> β*(_s * _s') + λ*I, s)

    return Σ
end

function make_distributions(K)
    μ = centeroids(K, a = -1.0)
    Σ = covmatrices(K, β=0.90, λ=1.0)

    D = [ MultivariateNormal(_μ, _Σ) for (_μ, _Σ) ∈ zip(μ, Σ) ]

    return D
end

##
K = 3 # Number of classes
N = 1000 # Number of samples per class

D = make_distributions(K);

# Data points
d = DataFrame()
for k ∈ 1:K
    global d
    data = rand(D[k], N)
    class = k * ones(Int64, N)

    d= vcat(d, DataFrame(:x=>data[1, :], :y=>data[2, :], :z=>class, copycols=false))
end

# Build linear models
full_model = lm(@formula(y ~ x), d)

# build conditional models
cond_models = map(k -> lm(@formula(y ~ x), d[d.z .== k, :]), 1:K)

## Plots distributions
fig = plot(size=(600, 600))

# Dsitribution
scatter!(fig, [_D.μ[1] for _D ∈ D], [_D.μ[2] for _D ∈ D], 
         markershape=:diamond, markersize=10, c=1:K, 
         label=nothing
)
for (i, _D) ∈ enumerate(D)
    covellipse!(fig, _D.μ, _D.Σ, n_std=1.96, show_axes=true, aspect_ratio=1, 
                # label=@sprintf("\$\\mu_%d = [%s], \\Sigma = [%s]\$", i,
                #   join([@sprintf("%+.2f", s) for s ∈ _D.μ], ", "),
                #   join([@sprintf("%+.2f", s) for s ∈ _D.Σ], ", ")
                # ),
                label=nothing,
                color=i, alpha=0.10
    )
end

# data points
@df d scatter!(:x, :y, color=:z, alpha=0.25, label=nothing)

# Plots linear models
Plots.abline!(reverse(coef(full_model))..., lw=3, label="\$\\mathcal{M}\$")
for k ∈ 1:K
    Plots.abline!(reverse(coef(cond_models[k]))..., lw=2, ls=:dash, label=@sprintf("\$\\mathcal{K}_%d\$", k))
end
plot!(title="Simpson's Paradox")

