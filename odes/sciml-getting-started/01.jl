# Source: https://docs.sciml.ai/SciMLSensitivity/stable/getting_started/#auto_diff
# Differentiating ODE solutions


using DifferentialEquations

##
function lotka_voltera!(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
end

##
p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0, 1.0]

##
prob = ODEProblem(lotka_voltera!, u0, (0.0, 10.0), p)
sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6)

##
using ForwardDiff

##
function f(x)
    _prob = remake(prob, u0=x[1:2], p=x[3:end])
    solve(_prob, Tsit5(), reltol=1e-6, abstol=1e-6, saveat=1)[1, :]
end

##
x = [u0; p]
dx = ForwardDiff.jacobian(f, x)

##
using Zygote, SciMLSensitivity

##
function sum_of_solution(u0, p)
    _prob = remake(prob, u0=u0, p=p)
    sum(solve(_prob, Tsit5(), reltol=1e-6, abstol=1e-6, saveat=0.1))
end

##
du01, dp1 = Zygote.gradient(sum_of_solution, u0, p)
