# Quick start - Bayesian coin flipping model

using Turing, StatsPlots, Random

##
p_true = 0.5

## Iterate from havinf seen 0 observations to 100 observations
Ns = 1:100

## Draw data from a Bernoulli distribution i.e. draw heads or tails
Random.seed!(12)
data = rand(Bernoulli(p_true), last(Ns));

## Declare Turing model
@model function coinflip(y)
    # Our prior belief about probability of heads in a coin
    p ~ Beta(1, 1)

    y .~ Bernoulli(p)
end

## Settings for the Hamiltorian Monte Carlo (HMC) sampler
iterations = 1000
ϵ = 0.05
τ = 10

## Start sampling
chain = sample(coinflip(data), HMC(ϵ, τ), iterations)

## Plot the sampling process for the parameter p, i.e. the probability of heads in the coin
histogram(chain[:p])
plot(chain)

