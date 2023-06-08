# Source: https://docs.sciml.ai/DiffEqDocs/stable/examples/conditional_dosing/#Conditional-Dosing-in-Pharmacometrics

# Conditional dosing in Pharmacometrics

using DifferentialEquations
using Plots

##
function f(du, u, p, t)
    println("p = ", p)
    du .= -u
end

##
u₀ = [10.0]
const V = 1
prob = ODEProblem(f, u₀, (0.0, 10.0))

##
sol = solve(prob, Tsit5())

##
plot(sol)

## ------------------
# condition(u, t, integrator) = t==4 && u[1]/V < 4
# affect!(integrator) = integrator.u[1] += 10 # Give new dose
# cb = DiscreteCallback(condition, affect!)

cb = DiscreteCallback(
    (u, t, integrator) -> t == 4 && u[1]/V < 4,
    integrator -> integrator.u[1] += 10
)

##
sol = solve(prob, Tsit5(), tstops=[4.0], callback=cb)
plot(sol)
