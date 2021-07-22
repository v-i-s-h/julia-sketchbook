# Infinite Mixture Models
# Source: https://turing.ml/dev/tutorials/06-infinite-mixture-model/

using Turing

##
@model function two_model(x)
    μ0 = 0.0
    σ1 = 1.0

    π1 ~ Beta(1, 1)
    π2 = 1 - π1

    μ1 ~ Normal(μ0, σ0)
    μ2 ~ Normal(μ0, σ0)

    z ~ Categorical([π1, π2])

    if z == 1
        x ~ Normal(μ1, 1.0)
    else
        x ~ Normal(μ2, 1.0)
    end
end

##
using Turing.RandomMeasures

##
# Concetration parameter
α = 10.0

# Random measure
rpm = DirichletProcess(α)

# Cluster assignments for each observation
z = Vector{Int}()

# Maximum number of observations we observe
Nmax = 500

for i ∈ 1:Nmax
    # Number of observations per cluster
    K = isempty(z) ? 0 : maximum(z)
    nk = Vector{Int}(map(k -> sum(z.==k), 1:K))

    push!(z, rand(ChineseRestaurantProcess(rpm, nk)))
end

##
using Plots
@gif for i ∈ 1:Nmax
    scatter(collect(1:i), z[1:i], markersize=2, xlabel="observation (i)", ylabel="cluster (k)", legend=false)
end

##
@model function infiniteGMM(x)
    α = 1.0
    μ0 = 0.0
    σ0 = 1.0

    rpm = DirichletProcess(α)

    H = Normal(μ0, σ0)


    z = tzeros(Int, length(x))

    μ = tzeros(Float64, 0)

    for i ∈ 1:length(x)
        K = maximum(z)
        nk = Vector{Int}(map(k -> sum(z.==k),1:K))

        z[i] ~ ChineseRestaurantProcess(rpm, nk)

        # Create new cluster?
        if z[i] > K
            push!(μ, 0.0)

            μ[z[i]] ~ H
        end

        x[i] ~ Normal(μ[z[i]], 1.0)
    end
end

##
using Plots, Random

Random.seed!(1)
data = vcat(randn(10), randn(10) .- 5, randn(10) .+ 10)
data .-= mean(data)
data /= std(data)

# MCMC sampling
Random.seed!(2)
iterations = 1000
model_fun = infiniteGMM(data)
chain = sample(model_fun, SMC(), iterations);

##
k = map(
    t -> length(unique(vec(chain[t, MCMCChains.namesingroup(chain, :z), :].value))),
    1:iterations
)

# visualize
plot(k, xlabel="Iteration", ylabel="Number of clusters", label="Chain 1")

##
histogram(k, xlabel="number of clusters", legend=false)
