# Bayesian Neural Network

using Lux
using Turing, Plots, Random, ReverseDiff, NNlib, Functors

Turing.setprogress!(false)
Turing.setadbackend(:reversediff)

## Generate data
N = 80
M = round(Int, N / 4)
rng = Random.default_rng()
Random.seed!(rng, 1234)

# Generate artificial data
x1s = rand(rng, Float32, M) * 4.5f0;
x2s = rand(rng, Float32, M) * 4.5f0;
xt1s = Array([[x1s[i] + 0.5f0; x2s[i] + 0.5f0] for i in 1:M])
x1s = rand(rng, Float32, M) * 4.5f0;
x2s = rand(rng, Float32, M) * 4.5f0;
append!(xt1s, Array([[x1s[i] - 5.0f0; x2s[i] - 5.0f0] for i in 1:M]))

x1s = rand(rng, Float32, M) * 4.5f0;
x2s = rand(rng, Float32, M) * 4.5f0;
xt0s = Array([[x1s[i] + 0.5f0; x2s[i] - 5.0f0] for i in 1:M])
x1s = rand(rng, Float32, M) * 4.5f0;
x2s = rand(rng, Float32, M) * 4.5f0;
append!(xt0s, Array([[x1s[i] - 5.0f0; x2s[i] + 0.5f0] for i in 1:M]))

# Store all the data for later
xs = [xt1s; xt0s]
ts = [ones(2 * M); zeros(2 * M)]

# Plot data points
function plot_data()
    x1 = first.(xt1s)
    y1 = last.(xt1s)
    x2 = first.(xt0s)
    y2 = last.(xt0s)

    plt = Plots.scatter(x1, y1; color="red", clim=(0, 1))
    Plots.scatter!(plt, x2, y2; color="blue", clim=(0, 1))

    return plt
end

plot_data()

## Building Neural Network
nn = Chain(Dense(2, 3, tanh), Dense(3, 2, tanh), Dense(2, 1, sigmoid))

ps, st = Lux.setup(rng, nn)

Lux.parameterlength(nn)

##
α = 0.09
sig = √(1.0 / α)

##
function vector_to_parameters(ps_new::AbstractVector, ps::NamedTuple)
    @assert length(ps_new) == Lux.parameterlength(ps)
    i = 1
    function get_ps(x)
        z = reshape(view(ps_new, i:(i+length(x) - 1)), size(x))
        i += length(x)
        return z
    end
    return fmap(get_ps, ps)
end

@model function bayes_nn(xs, ts, st)

    nparameters = Lux.parameterlength(nn)
    parameters ~ MvNormal(zeros(nparameters), sig .* ones(nparameters))

    preds, st = nn(xs, vector_to_parameters(parameters, ps), st)

    for i ∈ 1:length(ts)
        ts[i] ~ Bernoulli(preds[i])
    end
end

##
N = 5000
ch = sample(bayes_nn(hcat(xs...), ts, st), HMC(0.05, 4), N)

##
θ = MCMCChains.group(ch, :parameters).value;
