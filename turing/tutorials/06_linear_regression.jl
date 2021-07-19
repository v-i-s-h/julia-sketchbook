# Linear Regression
# Source: https://turing.ml/dev/tutorials/05-linear-regression/

using Turing, Distributions
using RDatasets
using MCMCChains, Plots, StatsPlots
using MLDataUtils: shuffleobs, splitobs, rescale!
using Distances

using Random
Random.seed!(0)

##
data = RDatasets.dataset("datasets", "mtcars")
first(data, 6)

## 
size(data)

## Data preprocessing
select!(data, Not(:Model))

trainset, testset = splitobs(shuffleobs(data), 0.70)

target = :MPG
train = Matrix(select(trainset, Not(target)))
test = Matrix(select(testset, Not(target)))
train_target = trainset[:, target]
test_target = testset[:, target]

μ, σ = rescale!(train; obsdim=1)
rescale!(test, μ, σ; obsdim=1)

μtarget, σtarget = rescale!(train_target; obsdim=1)
rescale!(test_target, μtarget, σtarget; obsdim=1);

##
# Bayesian Linear Regression
@model function linear_regression(x, y)
    σ₂ ~ truncated(Normal(0, 100), 0, Inf)

    intercept ~ Normal(0, √3)

    nfeatures = size(x, 2)
    coefficients ~ MvNormal(nfeatures, √10)

    mu = intercept .+ x * coefficients
    y ~ MvNormal(mu, √σ₂)
end

## 
model = linear_regression(train, train_target)
chain = sample(model, NUTS(0.65), 3_000)

##
plot(chain)

##
describe(chain)

## Comparing  to OLS
using GLM

train_with_intercept = hcat(ones(size(train, 1)), train)
ols = lm(train_with_intercept, train_target)

# Compute the predictions on train set and rescale them
p = GLM.predict(ols)
train_predictions_ols = μtarget .+ σtarget .* p

test_with_intercept = hcat(ones(size(test, 1)), test)
p = GLM.predict(ols, test_with_intercept)
test_prediction_ols = μtarget .+ σtarget .* p

## 
function prediction(chain, x)
    p = get_params(chain[200:end, :, :])
    targets = p.intercept' .+ x * reduce(hcat, p.coefficients)'

    return vec(mean(targets; dims=2))
end

##
p = prediction(chain, train)
train_predictions_bayes = μtarget .+ σtarget .* p
p = prediction(chain, test)
test_prediction_bayes = μtarget .+ σtarget .* p

##
DataFrame(
    MPG = testset[!, target],
    Bayes = test_prediction_bayes,
    OLS = test_prediction_ols
)

##
println(
    "Training set:",
    "\n\tBayes loss: ", msd(train_predictions_bayes, trainset[!, target]),
    "\n\tOLS loss: ", msd(train_predictions_ols, trainset[!, target])
)

println(
    "Test set:",
    "\n\tBayes loss: ", msd(test_prediction_bayes, testset[!, target]),
    "\n\tOLS loss: ", msd(test_prediction_ols, testset[!, target])
)