# Source: https://turing.ml/dev/tutorials/10-bayesian-differential-equations/

using Turing
using DifferentialEquations
using StatsPlots
using LinearAlgebra
using Random
Random.seed!(14);

## Define Lotka-Volterra model
function lotka_volterra(du, u, p, t)
    # Model parameters
    α, β, γ, δ = p
    # Current state
    x, y = u

    # Evaluate differnetial equations
    du[1] = (α - β * y) * x # prey
    du[2] = (δ * x - γ) * y # predator

    return nothing
end

## Define IVP and solve
u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 10.0)
prob = ODEProblem(lotka_volterra, u0, tspan, p)

# Solve
lv_sol = solve(prob, Tsit5())
plot(lv_sol)

## Create noisy data
sol = solve(prob, Tsit5(), saveat=0.1)
odedata = Array(sol) + 0.8 * randn(size(Array(sol)))

# Plot
plot(sol; alpha=0.3)
scatter!(sol.t, odedata'; label="")

## Bayesian Estimation using Turing
@model function fitlv(data, prob)
    # Prior distributions
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5, 0.5); lower=0.5, upper=2.5)
    β ~ truncated(Normal(1.2, 0.5); lower=0.0, upper=2.0)
    γ ~ truncated(Normal(3.0, 0.5); lower=1.0, upper=4.0)
    δ ~ truncated(Normal(1.0, 0.5); lower=0.0, upper=2.0)

    # Simulate Lotka-Volterra model
    p = [α, β, γ, δ]
    predicted = solve(prob, Tsit5(); p=p, saveat=0.1)

    # Observations
    for i ∈ 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ^2 * I)
    end

    return nothing
end

model = fitlv(odedata, prob);

## Sample
chain = sample(model, NUTS(0.65), MCMCSerial(), 1000, 3; progress=false)

## Plot chain
plot(chain)

## Data retrodiction
plot(; legend=false)
posterior_samples = sample(chain[[:α, :β, :γ, :δ]], 300; replace=false)
for p ∈ eachrow(Array(posterior_samples))
    sol_p = solve(prob, Tsit5(); p=p, saveat=0.1)
    plot!(sol_p; alpha=0.1, color="#BBBBBB")
end

plot!(sol; linewidth=1, color=[1, 2])
scatter!(sol.t, odedata'; color=[1, 2])

##

## Lotka-Volterra model without data of prey
@model function fitlv2(data::AbstractVector, prob)
    # Prior distributions
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5, 0.5); lower=0.5, upper=2.5)
    β ~ truncated(Normal(1.2, 0.5); lower=0.0, upper=2.0)
    γ ~ truncated(Normal(3.0, 0.5); lower=1.0, upper=4.0)
    δ ~ truncated(Normal(1.0, 0.5); lower=0.0, upper=2.0)

    # Simulate Lotka-Volterra model but save only the second state of the system
    p = [α, β, γ, δ]
    predicted = solve(prob, Tsit5(); p=p, saveat=0.1, save_idxs=2)

    # Observations of the predators
    data ~ MvNormal(predicted.u, σ^2 * I)

    return nothing
end

model2 = fitlv2(odedata[2, :], prob)
## Sample 3 independent chains
chain2 = sample(model2, NUTS(0.45), MCMCSerial(), 5000, 3; progress=false)

## Plot with data retrodiction
plot(; legend=false)
posterior_samples = sample(chain2[[:α, :β, :γ, :δ]], 300; replace=false)
for p ∈ eachrow(Array(posterior_samples))
    sol_p = solve(prob, Tsit5(); p=p, saveat=0.1)
    plot!(sol_p; α=0.1, color="#BBBBBB")
end
plot!(sol; color=[1 2], linewidth=1)
scatter!(sol.t, odedata'; color=[1 2])

## Inference of Delayed Differential Equations
function delayed_lotka_volterra(du, u, h, p, t)
    # model parameters
    α, β, γ, δ = p
    # Current state
    x, y = u

    # Evaluate differential equation
    du[1] = α * h(p, t-1; idxs=1) - β * x * y
    du[2] = -γ * y + δ * x * y

    return nothing
end

## Define IVP
p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
h(p, t; idxs::Int) = 1.0
prob_dde = DDEProblem(delayed_lotka_volterra, u0, h, tspan, p)

##
sol_dde = solve(prob_dde; saveat=0.1)
ddedata = Array(sol_dde) + 0.50 * randn(size(sol_dde))

## Plot simulation and noisy observations
plot(sol_dde)
scatter!(sol_dde.t, ddedata'; color=[1 2], label="")

##
@model function fitlv_dde(data, prob)
    # prior distributions
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5, 0.5); lower=0.5, upper=2.5)
    β ~ truncated(Normal(1.2, 0.5); lower=0.0, upper=2.0)
    γ ~ truncated(Normal(3.0, 0.5); lower=1.0, upper=4.0)
    δ ~ truncated(Normal(1.0, 0.5); lower=0.0, upper=2.0)

    # Simulate Lotka-Volterra
    p = [α, β, γ, δ]
    predicted = solve(prob, MethodOfSteps(Tsit5()); p=p, saveat=0.1)

    # Observations
    for i ∈ 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ^2 * I)
    end
end

##
model_dde = fitlv_dde(ddedata, prob_dde)

##
chain_dde = sample(model_dde, NUTS(0.65), MCMCSerial(), 300, 3; progress=false)

##
plot(chain_dde)

##
plot(; legend=false)
posterior_samples = sample(chain_dde[[:α, :β, :γ, :δ]], 300; replace=false)
for p ∈ eachrow(Array(posterior_samples))
    sol_p = solve(prob_dde, MethodOfSteps(Tsit5()); p=p, saveat=0.1)
    plot!(sol_p; alpha=0.1, color="#BBBBBB")
end
plot!(sol_dde; color=[1 2], linewidth=1)
scatter!(sol_dde.t, ddedata'; color=[1 2])

##

## #############################################################################
## Sensitivity analysis
using Zygote, SciMLSensitivity

##
# setadbackend(:zygote)
chain_zygote = sample(model, NUTS(0.65), 1000; progress=false)

##
