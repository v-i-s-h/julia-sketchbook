# Inifinite mixture models

using Turing
using Turing.RandomMeasures
using Plots
using FillArrays

## Two component model
@model function two_model(x)
    μ₀ = 0.0
    σ₀ = 1.0

    π₁ ~ Beta(1, 1)
    π₂ = 1 - π₁

    μ₁ ~ Normal(μ₀, σ₀)
    μ₂ ~ Normal(μ₁, σ₀)

    z ~ Categorical([π₁, π₂])

    if z == 1
        x ~ Normal(μ₁, 1.0)
    else
        x ~ Normal(μ₂, 1.0)
    end
end

## Infinite mixture model
α = 10.0
rpm = DirichletProcess(α)
z = Vector{Int}()
Nmax = 500

for i ∈ 1:Nmax
    K = isempty(z) ? 0 : maximum(z)
    nk = Vector{Int}(map(k -> sum(z .== k), 1:K))

    # Draw new assignment
    push!(z, rand(ChineseRestaurantProcess(rpm, nk)))
end

##
@gif for i ∈ 1:Nmax
    scatter(
        collect(1:i),
        z[1:i],
        markersize=2,
        xlabel="oᵢ",
        ylabel="kᵢ",
        legend=false
    )
end

## 
@model function infiniteGMM(x)
    α = 1.0
    μ₀ = 0.0
    σ₀ = 1.0

    rpm = DirichletProcess(α) # 
    H = Normal(μ₀, σ₀) # Base measure
    z = tzeros(Int, length(x)) # Latent assignment
    μ = tzeros(Float64, 0) # Location of clusters

    for i ∈ 1:length(x)
        K = maximum(z)
        nk = Vector{Int}(map(k -> sum(z .== k), 1:K))

        z[i] ~ ChineseRestaurantProcess(rpm, nk)

        if z[i] > K
            push!(μ, 0.0)
            μ[z[i]] ~ H
        end

        x[i] ~ Normal(μ[z[i]], 1.0)
    end
end

##
data = vcat(randn(10), randn(10) .- 5, randn(10) .+ 10)
data .-= mean(data)
data /= std(data)

##
iterations = 1000
model_fn = infiniteGMM(data)
chain = sample(model_fn, SMC(), iterations)

##
k = map(
    t -> length(unique(vec(chain[t, MCMCChains.namesingroup(chain, :z), :].value))),
    1:iterations
)
plot(k; xlabel="Iterations", ylabel="number of clusters", label="Chain 1")

##
histogram(k; xlabel="Number of clusters", legend=false)

##