# Bayesian neural network using Flux and Turing
# Source: http://dm13450.github.io/2020/12/19/EdwardAndFlux.html

using Flux
using Plots
using Distributions

##
f(x) = cos(x) + rand(Normal(0, 0.1))

##
xTrain = collect(-3.0:0.1:+3.0)
yTrain = f.(xTrain)
plot(xTrain, yTrain, seriestype=:scatter, label="Train data")
plot!(xTrain, cos.(xTrain), label="Truth")

##
model = Chain(Dense(1, 2, tanh), Dense(2, 1))
loss(x, y) = Flux.Losses.mse(model(x), y)
optmiser = Descent(0.1)

##
x = rand(Normal(), 100)
y = f.(x)
train_data = Iterators.repeated((Array(x'), Array(y')), 100)

##
Flux.@epochs 10 Flux.train!(loss, Flux.params(model), train_data, optmiser)

##
yOut = zeros(length(xTrain))
for (i, x) in enumerate(xTrain)
    yOut[i] = model([xTrain[i]])[1]
end

plot(xTrain, yOut, label="Predicted")
plot!(xTrain, cos.(xTrain), label="True")
plot!(x, y, seriestype=:scatter, label="Data")

##
using Turing

##
function unpack(nn_params::AbstractVector)
    # print(size(nn_params))
    W₁ = reshape(nn_params[1:2], 2, 1)
    b₁ = reshape(nn_params[3:4], 2)

    W₂ = reshape(nn_params[5:6], 1, 2)
    b₂ = [nn_params[7]]

    return W₁, b₁, W₂, b₂
end

function nn_forward(xs, nn_params::AbstractVector)
    W₁, b₁, W₂, b₂ = unpack(nn_params)
    nn = Chain(Dense(W₁, b₁, tanh), Dense(W₂, b₂))

    return nn(xs)
end

##
α = 0.1

σ = √(1.0 / α)

@model bayes_nn(xs, ys) = begin
    nn_params ~ MvNormal(zeros(7), σ .* ones(7)) # Prior

    preds = nn_forward(xs, nn_params) # build the net
    sigma ~ Gamma(0.01, 1/0.01) # Prior for the variance
    for i = 1:length(ys)
        ys[i] ~ Normal(preds[i], sigma)
    end
end

## 
N = 5000
ch1 = sample(bayes_nn(hcat(x...), y), NUTS(0.65), N)
ch2 = sample(bayes_nn(hcat(x...), y), NUTS(0.65), N)

##
lp, maxInd = findmax(ch1[:lp])

params, _ = ch1.name_map
bestParams = map(x -> ch1[x].data[maxInd], params[1:7])
plot(x, cos.(x), seriestype=:line, label="True")
plot!(x, Array(nn_forward(hcat(x...), bestParams)'),
      seriestype=:scatter, label="MAP Estimate")

## 
xPlot = sort(x)

sp = plot()

for i in max(1, (maxInd[1]-100)):min(N, (maxInd[1]+100))
    parSample = map(x -> ch1[x].data[i], params)
    plot!(sp, xPlot, Array(nn_forward(hcat(xPlot...), parSample)'),
        label=:none, color="blue")
end

plot!(sp, x, y, seriestype=:scatter, label="Training Data", color="red")

##
lPlot = plot(ch1[:lp], label="Chain 1", title="Log posterior")
plot!(lPlot, ch2[:lp], label="Chain 2")

sigPlot = plot(ch1[:sigma], label="Chain 1", title="Variance")
plot!(sigPlot, ch2[:sigma], label="Chain 2")

plot(lPlot, sigPlot)