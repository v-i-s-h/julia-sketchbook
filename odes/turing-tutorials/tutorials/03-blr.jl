# Bayesian Logistic Regression

using Turing, Distributions
using RDatasets
using MCMCChains, Plots, StatsPlots
using StatsFuns: logistic
using MLDataUtils: shuffleobs, stratifiedobs, rescale!

using Random
Random.seed!(0)
Turing.setprogress!(false);

## Data cleaning and setup
data = RDatasets.dataset("ISLR", "Default");
first(data, 6)

## Convert categorial to numerical values
data[!, :DefaultNum] = [r.Default == "Yes" ? 1.0 : 0.0 for r in eachrow(data)]
data[!, :StudentNum] = [r.Student == "Yes" ? 1.0 : 0.0 for r in eachrow(data)]
select!(data, Not([:Default, :Student]))
first(data, 6)

## split data standardize
function split_data(df, target; at=0.70)
    shuffled = shuffleobs(df)
    return trainset, testset = stratifiedobs(row -> row[target], shuffled; p=at)
end

features = [:StudentNum, :Balance, :Income]
numerics = [:Balance, :Income]
target = :DefaultNum

trainset, testset = split_data(data, target; at=0.05)
for feature ∈ numerics
    μ, σ = rescale!(trainset[!, feature]; obsdim=1)
    rescale!(testset[!, feature], μ, σ; obsdim=1)
end

# Convert to matrix format
train = Matrix(trainset[:, features])
test = Matrix(testset[:, features])
train_label = trainset[:, target]
test_label = testset[:, target];

## Model declaration
@model function logistic_regression(x, y, n, σ)
    intercept ~ Normal(0, σ)

    student ~ Normal(0, σ)
    balance ~ Normal(0, σ)
    income ~ Normal(0, σ)

    # for i ∈ 1:n
    #     v = logistic(intercept + 
    #                     student * x[i, 1] + 
    #                     balance * x[i, 2] +
    #                     income * x[i, 3])
    #     y[i] ~ Bernoulli(v)
    # end

    y .~ Bernoulli.(logistic.(x * [student, balance, income] .+ intercept))
end

## Sampling
n = size(train, 1)
m = logistic_regression(train, train_label, n, 1)
chain = sample(m, HMC(0.05, 10), MCMCThreads(), 1500, 3)

## 
plot(chain)

##
l = [:student, :balance, :income]
corner(chain, l)

## Make predictions
function prediction(x::Matrix, chain, threshold)
    intercept = mean(chain[:intercept])
    student = mean(chain[:student])
    balance = mean(chain[:balance])
    income = mean(chain[:income])

    n = size(x, 1)

    v = Vector{Float64}(undef, n)

    for i ∈ 1:n
        num = logistic(intercept + student * x[i, 1] + 
                    balance * x[i, 2] + income * x[i, 3])
        if num > threshold
            v[i] = 1
        else
            v[i] = 0
        end
    end
    
    return v
end

## Make predictions and test
threshold = 0.07
predictions = prediction(test, chain, threshold)
loss = sum((predictions - test_label) .^ 2) / length(test_label)

##
defaults = sum(test_label)
not_defaults = length(test_label) - defaults

predicted_defaults = sum(test_label .== predictions .== 1)
predicted_not_defaults = sum(test_label .== predictions .== 0)

println("Defaults: $defaults
    Predictions: $predicted_defaults
    Percentage defaults correct $(predicted_defaults / defaults)")
println("Not  defaults: $not_defaults
    Preditions: $predicted_not_defaults
    Percentage not-defaults correct $(predicted_not_defaults / not_defaults)")

