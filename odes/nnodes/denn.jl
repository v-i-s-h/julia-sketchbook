# Source: https://julialang.org/blog/2019/01/fluxdiffeq/

##

using DifferentialEquations
using Plots

##
function lotka_volterra(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α*x - β*x*y
    du[2] = -δ*y + γ*x*y
end

##
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
p = [1.5, 1.0, 3.0, 1.0]

##
prob = ODEProblem(lotka_volterra, u0, tspan, p)

##
sol = solve(prob)

## 
plot(sol)

## Create data points
p = [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(lotka_volterra, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=0.1)
A = sol[1, :]

##
plot(sol)
scatter!(sol.t, A)

## NN ODEs
using Flux, DiffEqFlux

##
p = [2.2, 1.0, 2.0, 0.4] # initial params for model
params = Flux.params(p)

function predict_rd() # 1-layer NN
    solve(prob, Tsit5(), p=p, saveat=0.1)[1, :]
end

loss_rd() = sum(abs2, x-1 for x ∈ predict_rd()) # loss function

##
data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
    display(loss_rd())
    display(plot(solve(remake(prob, p=p), Tsit5(), saveat=0.1), ylim=(0, 6)))
end

##
cb() # display the ODE with initial parameters

## Train the model
Flux.train!(loss_rd, params, data, opt, cb=cb)

## Multi-layer


## Suite
using ParameterizedFunctions

##

rober = @ode_def Rober begin
    dy₁ = -k₁*y₁ + k₃*y₂*y₃
    dy₂ = k₁*y₁ - k₂*y₂^2 - k₃*y₂*y₃
    dy₃ = k₂*y₂^2
end k₁ k₂ k₃
prob = ODEProblem(rober, [1.0; 0.0; 0.0], (0.0, 1e11), (0.04, 3e7, 1e4))
solve(prob, KenCarp4())

##

## Neural ODE layer
dudt = Chain(Dense(2, 50, tanh), Dense(50, 2))
tspan = (0.0f0, 25.0f0)
node = NeuralODE(dudt, tspan, Tsit5(), saveat=0.1)


## Create data from an ODE
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1], tspan[2], length=datasize)
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob, Tsit5(), saveat=t))

## Create NN
dudt = Chain(
            x -> x.^3,
            Dense(2, 50, tanh),
            Dense(50, 2)
        )
n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
ps = Flux.params(n_ode)

##
pred = n_ode(u0) # Get prediction using correct initial condition
scatter(t, ode_data[1, :], label="data")
scatter!(t, pred[1, :], label="prediction")

##
function predict_n_ode()
    n_ode(u0)
end
loss_n_ode() = sum(abs2, ode_data .- predict_n_ode())

##
data = Iterators.repeated((), 1000)
opt = ADAM(0.1)
cb = function()
    display(loss_n_ode())
    curr_pred = predict_n_ode()
    pl = scatter(t, ode_data[1, :], label="data")
    scatter!(pl, t, curr_pred[1, :], label="prediction")
    display(plot(pl))
end

##
cb()

##
Flux.train!(loss_n_ode, ps, data, opt, cb=cb)