# Linear Regression

using Turing, Distributions
using RDatasets
using MCMCChains, Plots, StatsPlots
using MLDataUtils: shuffleobs, splitobs, rescale!
using Distances
using FillArrays
using LinearAlgebra

# using Random; Random.seed!(42);
Turing.setprogress!(false);

## Load data
data = RDatasets.dataset("datasets", "mtcars")
first(data, 6)

## Data preprocessing
select!(data, Not(:Model))
trainset, testset = splitobs(shuffleobs(data), 0.7)

target = :MPG
train = Matrix(select(trainset, Not(target)))
test = Matrix(select(testset, Not(target)))
train_target = trainset[:, target]
test_target = testset[:, target]

μ, σ = rescale!(train; obsdim=1)
rescale!(test, μ, σ; obsdim=1)

μₜ, σₜ = rescale!(train_target; obsdim=1)
rescale!(test_target, μₜ, σₜ; obsdim=1)

## Model specification - Bayesian linear regression
@model function linear_regression(x, y)
    # variance prior
    σ² ~ truncated(Normal(0, 100); lower=0)

    # intercept prior
    α ~ Normal(0, √3)

    # Priors on coefficiets
    nfeatures = size(x, 2)
    β ~ MvNormal(Zeros(nfeatures), 10.0 * I)

    # Caculate all mu terms
    μ = α .+ x * β

    return y ~ MvNormal(μ, σ²)
end

##
model = linear_regression(train, train_target)
chain = sample(model, NUTS(0.65), 3000)

##
plot(chain)

## Make predictions with input vector
function prediction(chain, x)
    p = get_params(chain[200:end, :, :])
    predictions = p.α' .+ x * reduce(hcat, p.β)'
    return vec(mean(predictions; dims=2))
end

p = prediction(chain, train)
train_prediction_bayes = μₜ .+ σₜ .* p
p = prediction(chain, test)
test_prediction_bayes = μₜ .+ σₜ .* p
 
## Compare to OLS
using GLM

train_with_intercept = hcat(ones(size(train, 1)), train)
ols = lm(train_with_intercept, train_target)

p = GLM.predict(ols)
train_prediction_ols = μₜ .+ σₜ .* p

test_with_intercept = hcat(ones(size(test, 1)), test)
p = GLM.predict(ols, test_with_intercept)
test_prediction_ols = μₜ .+ σₜ .* p

## Results table
results = DataFrame(; MPG=testset[!, target], 
                      Bayes=test_prediction_bayes,
                      OLS=test_prediction_ols)

##
println(
    "Training set:",
    "\n\tBayes loss: ",
    msd(train_prediction_bayes, trainset[!, target]),
    "\n\tOLS loss: ",
    msd(train_prediction_ols, trainset[!, target]),
)

println(
    "Test set:",
    "\n\tBayes loss: ",
    msd(test_prediction_bayes, testset[!, target]),
    "\n\tOLS loss: ",
    msd(test_prediction_ols, testset[!, target]),
)
