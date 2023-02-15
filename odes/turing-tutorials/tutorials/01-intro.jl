# Introduction to Turing

using Distributions
using Random
using StatsPlots

Random.seed!(12);

## Generate data
p₀ = 0.50;
N = 100;
data = rand(Bernoulli(p₀), N);

## Define prior
prior_belief = Beta(1, 1);

## Update posterior
function updated_belief(prior_belief::Beta, data::AbstractArray{Bool})
    heads = sum(data)
    tails = length(data) - heads

    # Update prior_belief
    return Beta(prior_belief.α + heads, prior_belief.β + tails)
end

## Show updated belief for an increasing number of observations
@gif for n ∈ 0:N
    plot(
         updated_belief(prior_belief, data[1:n]);
         size=(500, 250),
         title="Updated belief after $n observations",
         ylabel="",
         legend=nothing,
         xlim=(0, 1),
         fill=0,
         α=0.3,
         w=3,
    )
    vline!([p₀])
end

##
# Coin flipping with Turing
##

using Turing

## Define coin flip model
@model function coin_flip(; N::Int)
    # Prior belief
    p ~ Beta(1, 1)

    y ~ filldist(Bernoulli(p), N)

    return y
end

## random samples from prior model
rand(coin_flip(; N))

## Model can be conditioned on observations
coin_flip(y::AbstractVector{<:Real}) = coin_flip(; N=length(y)) | (; y)
model = coin_flip(data);

## define sampler
sampler = HMC(0.05, 10)

## Sample
chain = sample(model, sampler, 1000; progress=false)

## Visulize results
histogram(chain)


