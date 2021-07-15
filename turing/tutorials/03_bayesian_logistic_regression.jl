# using Base: Float16, Float64
# using DataFrames: Matrix
# Bayesian Logistic Regression
# Source: https://turing.ml/dev/tutorials/02-logistic-regression/

# Import Turing ad Distributions
using Turing, Distributions

# Import Rdatasets
using RDatasets
# using DataFrames

using MCMCChains, Plots, StatsPlots

# We need a logistic function, which is provided by StatsFun
using StatsFuns: logistic

# Functionality for splitting and normalizing the data
using MLDataUtils: shuffleobs, stratifiedobs, rescale!

# Set a seed for reproducibility
using Random
# Random.seed!(0)

##
data = RDatasets.dataset("ISLR", "Default")

first(data, 6)

## Data preprocssing
data[!, :DefaultNum] = [r.Default == "Yes" ? 1.0 : 0.0 for r in eachrow(data)]
data[!, :StudentNum] = [r.Student == "Yes" ? 1.0 : 0.0 for r in eachrow(data)]

select!(data, Not([:Default, :Student]))

## 
first(data, 6)

##
function split_data(df, target; at=0.70)
    shuffled = shuffleobs(df)
    trainset, testset = stratifiedobs(row -> row[target], shuffled, p=at)
end

features = [:StudentNum, :Balance, :Income]
numerics = [:Balance, :Income]
target = :DefaultNum

trainset, testset = split_data(data, target, at=0.05)
for feature in numerics
    μ, σ = rescale!(trainset[!, feature], obsdim=1)
    rescale!(testset[!, feature], μ, σ, obsdim=1)
end

train = Matrix(trainset[:, features])
test = Matrix(testset[:, features])
train_label = trainset[:, target]
test_label = testset[:, target];

##
# Bayesian Logistic Regression (LR)
@model logistic_regression(x, y, n, σ) = begin
    intercept ~ Normal(0, σ)

    student ~ Normal(0, σ)
    balance ~ Normal(0, σ)
    income ~ Normal(0, σ)

    for i ∈ 1:n
        v = logistic(intercept + student * x[i, 1] + balance * x[i, 2] + balance * x[i, 3])
        y[i] ~ Bernoulli(v)
    end
end

##
# Sampling
n, _ = size(train)

chain = mapreduce(
    c -> sample(logistic_regression(train, train_label, n, 1),
        HMC(0.05, 10),
        1500),
    chainscat,
    1:3
)

describe(chain)

##
plot(chain)

##
l = [:student, :balance, :income]
corner(chain, l)

##
# Making predictions
function prediction(x::Matrix, chain, threshold)
    # Pull the means from each parameter's sampled values in the chain
    intercept = mean(chain[:intercept])
    student = mean(chain[:student])
    balance = mean(chain[:balance])
    income = mean(chain[:income])

    # Retrieve the number of rows
    n, _ = size(x)

    # Generate a vector to store our predictions
    v = Vector{Float64}(undef, n)

    # Calculate the logistic function for each element in the test set
    for i ∈ 1:n
        num = logistic(intercept .+ student * x[i, 1] + balance * x[i, 2] + income * x[i, 3])
        if num >= threshold
            v[i] = 1
        else
            v[i] = 0
        end
    end
    return v
end

##
# Set the prediction threshold
threshold = 0.07

# Make the predictions
predictions = prediction(test, chain, threshold)

# calculate MSE for our test set
loss = sum((predictions - test_label).^2) / length(test_label)

##
defaults = sum(test_label)
not_defaults = length(test_label) - defaults

predicted_defaults = sum(test_label .== predictions .== 1)
predicted_not_defaults = sum(test_label .== predictions .== 0)

println("Defaults: $defaults
    Predictions: $predicted_defaults
    Prectange defaults correct: $(predicted_defaults/defaults)")

println("Not Defaults: $not_defaults
    Predictions: $predicted_not_defaults
    Percentage non-defaults correct $(predicted_not_defaults/not_defaults)")
