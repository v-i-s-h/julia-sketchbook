# Source: https://docs.sciml.ai/DiffEqDocs/stable/getting_started/#Example-3:-Solving-Nonhomogeneous-Equations-using-Parameterized-Functions
# Pendulum


using DifferentialEquations
using Plots

##
l = 1.0
m = 1.0
g = 9.81

##
function pendulum!(du, u, p, t)
    du[1] = u[2]
    du[2] = -3g/(2l) * sin(u[1]) + 3/(m*l^2) * p(t)
end

##
θ₀ = 0.01
ω₀ = 0.0
u₀ = [θ₀, ω₀]
tspan = (0.0, 10.0)

M = t -> 0.1 * sin(t)

prob = ODEProblem(pendulum!, u₀, tspan, M)
sol = solve(prob)

##
plot(sol, lw=2, xaxis="t", label=["θ [rad]" "ω[rad/s]"], layout=(2, 1))

