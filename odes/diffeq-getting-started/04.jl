# Source: https://docs.sciml.ai/DiffEqDocs/stable/examples/classical_physics/#First-order-linear-ODE

# First Order linear ODE

using OrdinaryDiffEq
using Plots

##
C₁ = 5.730

##
u₀ = 1.0
tspan = (0.0, 1.0)

##
radioactivedecay(u, p, t) = -C₁ * u

##
prob = ODEProblem(radioactivedecay, u₀, tspan)
sol = solve(prob, Tsit5())

##
plot(title="Carbon half life", xaxis="Time in thousands of years", yaxis="Percentage left")
plot!(sol, lw=2, label="Numerical Solution")
plot!(sol.t, t -> exp(-C₁ * t), lw=2, label="Analytical Solving")

