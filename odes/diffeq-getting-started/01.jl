# https://docs.sciml.ai/DiffEqDocs/stable/getting_started/

##

using DifferentialEquations
using Plots

##
α = 1.01
f(u, p, t ) = α * u

##
u0 = 0.5
tspan = (0.0, 1.0)
prob = ODEProblem(f, u0, tspan)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

##
plot(title="Solution of linear ODE", xaxis="Time(t)", y = "u(t) (in μm)")
plot!(sol, linewidth=2, label="Computed solution")
plot!(sol.t, t -> 0.50 * exp(α*t), lw=2, ls=:dash, label="True solution")

##
