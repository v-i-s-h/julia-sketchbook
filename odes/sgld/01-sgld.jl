# Source: https://sebastiancallh.github.io/post/langevin/

using RDatasets, Statistics
using Plots, StatsPlots, Random
using MLDataUtils: stratifiedobs, rescale!
using Flux

Random.seed!(0)

# device
device = cpu

##
data = dataset("ISLR", "Default")
todigit(x) = x == "Yes" ? 1.0 : 0.0
data[!, :Default] = map(todigit, data[!, :Default])
data[!, :Student] = map(todigit, data[!, :Student])
println("Data size:", size(data))
first(data, 5)

##
target = :Default
numerics = [:Balance, :Income]
features = [:Student, :Balance, :Income]
train, test = stratifiedobs(row -> row[target], data, p=0.70, shuffle=true)


for feature ∈ numerics
    μ, σ = rescale!(train[!, feature]; obsdim=1)
    rescale!(test[!, feature], μ, σ; obsdim=1)
end

##
x_train = Matrix(train[:, features])' |> device
y_train = reshape(train[:, target], 1, :) |> device
x_test = Matrix(test[:, features])' |> device
y_test = reshape(test[:, target]', 1, :) |> device

##
function train_logreg(; steps, update)
    Random.seed!(1)

    paramvec(θ) = reduce(hcat, θ |> cpu)
    
    model = Dense(length(features), 1, sigmoid) |> device
    θ = Flux.params(model)
    θ₀ = paramvec(θ)

    predict(x; τ=0.50) = model(x) .> τ
    accuracy(x, y) = mean(cpu(predict(x)) .== cpu(y))

    loss(x, y) = mean(Flux.binarycrossentropy.(model(x), y))
    trainloss() = loss(x_train, y_train)
    testloss() = loss(x_test, y_test)

    trainlosses = [cpu(trainloss()); zeros(steps)]
    testlosses = [cpu(testloss()); zeros(steps)]
    weights = [cpu(θ₀); zeros(steps, length(θ₀))]

    for t ∈ 1:steps
        ∇L = gradient(trainloss, θ)
        foreach(θᵢ -> update(∇L, θᵢ, t), θ)

        weights[t+1, :] = paramvec(θ) |> cpu
        trainlosses[t+1] = trainloss() |> cpu
        testlosses[t+1] = testloss() |> cpu
    end

    println("Final parameters are $(paramvec(θ))")
    println("Test accuracy is $(accuracy(x_test, y_test))")

    return model, weights, trainlosses, testlosses
end

##
sgd(∇L, θᵢ, t, η=2) = begin
    Δθᵢ = η .* ∇L[θᵢ]
    θᵢ .-= Δθᵢ
end

##
model, weights, trainlosses, testlosses = train_logreg(; steps=1000, update=sgd)

## Plot results
p = plot(size=(800, 400))
f = [features..., :intercept]
for (i, _f) ∈ enumerate(f)
    plot!(p, weights[:, i], label=String(_f))
end
plot(p)
## Bayesian Logistics Regression
sgld(∇L, θᵢ, t, a=10.0, b=1000, γ=0.9) = begin
    ϵ = a * (b + t) ^ -γ # noise decay
    η = ϵ .* randn(size(θᵢ)) |> gpu # random noise
    Δθᵢ = 0.5ϵ * ∇L[θᵢ] + η # step
    θᵢ .-= Δθᵢ
end

##
model_sgld, weights_sgld, trainlosses_sgld, testlosses_sgld = train_logreg(steps=20000, update=sgld)

##
p = plot(size=(800, 400))
f = [features..., :intercept]
for (i, _f) ∈ enumerate(f)
    plot!(p, weights_sgld[:, i], label=String(_f))
end
plot(p)

## Plot density of weights
plots = []
f = [features..., :intercept]
for (i, _f) ∈ enumerate(f)
    p = plot()
    density!(p, weights[:, i], label="GD")
    density!(p, weights_sgld[end-2000:end, i], label="SGLD")
    plot!(title=label=String(_f))
    plots = [plots..., p]
end
plot(plots...; size=(800, 800))
