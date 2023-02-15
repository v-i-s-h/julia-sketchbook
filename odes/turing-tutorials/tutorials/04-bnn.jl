# Bayesian Neural Networks

using Turing, Flux, Plots, Random, ReverseDiff

Turing.setprogress!(false)
Turing.setadbackend(:reversediff)

## Generate synthetic dataset
N = 80
M = round(Int, N/4)
Random.seed!(1234)

x1s = rand(M) * 4.5
x2s = rand(M) * 4.5
xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i ∈ 1:M])
x1s = rand(M) * 4.5
x2s = rand(M) * 4.5
append!(xt1s, Array([[x1s[i] - 5; x2s[i] - 5] for i ∈ 1:M]))

x1s = rand(M) * 4.5
x2s = rand(M) * 4.5
xt0s = Array([[x1s[i] + 0.5; x2s[i] - 5] for i ∈ 1:M])
x1s = rand(M) * 4.5
x2s = rand(M) * 4.5
append!(xt0s, Array([[x1s[i] - 5; x2s[i] + 0.5] for i ∈ 1:M]))

xs = [xt1s; xt0s];
ts = [ones(2*M); zeros(2 * M)];

# Plot data points
function plot_data()
    x1 = map(e -> e[1], xt1s); y1 = map(e -> e[2], xt1s);
    x2 = map(e -> e[1], xt0s); y2 = map(e -> e[2], xt0s);

    scatter(x1, y1; color="red", clim=(0, 1))
    scatter!(x2, y2; color="blue", clim=(0, 1))
end

plot_data()

## Building a neural network
nn_initial = Chain(
    Dense(2, 3, tanh),
    Dense(3, 2, tanh),
    Dense(2, 1, σ)
)
parameters_initial, reconstruct = Flux.destructure(nn_initial)

length(parameters_initial)

## Create probabilitic model
alpha = 0.09
sig = √(1 / alpha)

@model function bayes_nn(xs, ts, nparameters, reconstruct)
    parameters ~ MvNormal(zeros(nparameters), sig .* ones(nparameters))

    nn = reconstruct(parameters)
    preds = nn(xs)

    for i ∈ axes(ts, 1)
        ts[i] ~ Bernoulli(preds[i])
    end
end

## Perform inference
N = 5000
ch = sample(
    bayes_nn(hcat(xs...), ts, length(parameters_initial), reconstruct),
    HMC(0.05, 4),
    N
);

## Extract sampled weights and bias parameters
θ = MCMCChains.group(ch, :parameters).value;

## MAP estimation
nn_forward(x, θ) = reconstruct(θ)(x)

plot_data()

# Find index that provided highest log posterior in the chain
_, i = findmax(ch[:lp])
i = i.I[1] # extract index

x1_range = collect(range(-6; stop=6, length=25))
x2_range = collect(range(-6; stop=6, length=25))
Z = [nn_forward([x1, x2], θ[i, :])[1] for x1 ∈ x1_range, x2 ∈ x2_range]
contour!(x1_range, x2_range, Z)

## BMA
function nn_predict(x, θ, num)
    return mean([nn_forward(x, θ[i, :]) for i ∈ 1:10:num])
end

plot_data()

n_end = 1500
x1_range = collect(range(-6; stop=6, length=25))
x2_range = collect(range(-6; stop=6, length=25))
Z = [nn_predict([x1, x2], θ, n_end)[1] for x1 ∈ x1_range, x2 ∈ x2_range]
contour!(x1_range, x2_range, Z)

## Plot evolution
n_end = 500
anim = @gif for i in 1:n_end
    plot_data()
    Z = [nn_forward([x1, x2], θ[i, :])[1] for x1 in x1_range, x2 in x2_range]
    contour!(x1_range, x2_range, Z; title="Iteration $i", clim=(0, 1))
end every 5
