# Simple example

using Turing
using StatsPlots

## Define a simple model with unknown mean and variance
@model function gdemo(x, y)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    y ~ Normal(m, sqrt(s²))
end

## Run sampler, collect results
chain = sample(gdemo(1.5, 2), HMC(0.1, 5), 1000)

## Summarize results
describe(chain)

## Plot results
p = plot(chain)


## Sample from prior
model = gdemo(1.5, 2)
prior_chain = sample(model, Prior(), 100)

## Compute MLE/MAP estimates
using Optim

##
optimize(model, MLE())

##
optimize(model, MAP())

##

