# Bayesian Estimation of Differential Equations

using Turing
using DifferentialEquations
using StatsPlots
using LinearAlgebra

using Random; Random.seed!(14);

##
function lotka_volterra(du, u, p, t)
    α, β, γ, δ = p
    x, y = u

    du[1] = (α - β * y) * x
    du[2] = (δ * x - γ) * y

    return nothing
end

## Define IVP
u₀ = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 10.0)
prob = ODEProblem(lotka_volterra, u₀, tspan, p)

##
plot(solve(prob, Tsit5()))

## Generate noisy observations
sol = solve(prob, Tsit5(); saveat=0.1)
odedata = Array(sol) + 0.80 * randn(size(Array(sol)))

plot(sol; alpha=0.3)
scatter!(sol.t, odedata'; color=[1 2], label="")

## Bayesian estimation
@model function fitlv(data, prob)
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5, 0.5); lower=0.5, upper=2.5)
    β ~ truncated(Normal(1.2, 0.5); lower=0.0, upper=2.0)
    γ ~ truncated(Normal(3.0, 0.5); lower=1.0, upper=4.0)
    δ ~ truncated(Normal(1.0, 0.5); lower=0.0, upper=2.0)

    # Simulate Lotka-Volterra model
    p = (α, β, γ, δ)
    predicted = solve(prob, Tsit5(); p=p, saveat=0.1)

    # observations
    for i ∈ 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ^2 * I)
    end
    
    return nothing
end

model = fitlv(odedata, prob)

##
chain = sample(model, NUTS(0.65), MCMCSerial(), 1500, 3)

##
plot(chain)

## data retrodiction
plot(; legend=false)
posterior_samples = sample(chain[[:α, :β, :γ, :δ]], 300; replace=false)
for p ∈ eachrow(Array(posterior_samples))
    sol_p = solve(prob, Tsit5(); p=p, saveat=0.1)
    plot!(sol_p; alpha=0.1, color="#BBBBBB")
end
plot!(sol; color=[1 2], linewidth=1)
scatter!(sol.t, odedata'; color=[1 2])


## Scaling to large models
using Zygote, SciMLSensitivity
setadbackend(:zygote)
sample(model, NUTS(0.65), 1000; progress=false)

##
@model function fitlv_sensealg(data, prob)
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5, 0.5); lower=0.5, upper=2.5)
    β ~ truncated(Normal(1.2, 0.5); lower=0.0, upper=2.0)
    γ ~ truncated(Normal(3.0, 0.5); lower=1.0, upper=4.0)
    δ ~ truncated(Normal(1.0, 0.5); lower=0.0, upper=2.0)

    p = [α, β, γ, δ]
    predicted = solve(prob; p=p, saveat=0.1)

    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ^2 * I)
    end

    return nothing
end

model_sensealg = fitlv_sensealg(odedata, prob)

setadbackend(:zygote)
sample(model_sensealg, NUTS(0.65), 1000; progress=false)

##