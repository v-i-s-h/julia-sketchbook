# Source: https://docs.sciml.ai/DiffEqDocs/stable/examples/classical_physics/#Second-Order-Linear-ODE

# Simple harmonic oscillator

using OrdinaryDiffEq
using Plots

##
ω = 1

##
x₀ = [0.0]
dx₀ = [π/2]
tspan = (0.0, 2π)

ϕ = atan((dx₀[1]/ω) / x₀[1])
A = √(x₀[1]^2 + dx₀[1]^2)

##
function harmonicoscillator!(ddu, du, u, ω, t)
    ddu .= -ω^2 * u
end

##
prob = SecondOrderODEProblem(harmonicoscillator!, dx₀, x₀, tspan, ω)
sol = solve(prob, DPRKN6())

##
plot(titile="Simple Harmic Oscillator", xaxis="Time", yaxis="\$x\$")
plot!(sol, idxs=[2, 1], label=["x" "dx"])
plot!(t -> A * cos(ω*t - ϕ), lw=2, ls=:dash, label="x (analytical)")
plot!(t -> -A * ω * sin(ω*t - ϕ), lw=2, ls=:dash, label="dx (analytical)")
