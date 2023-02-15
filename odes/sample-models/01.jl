# A sample model for ODE data generation

using DifferentialEquations
using Plots

## Define model
function wave_model!(du, u, p, t)
    ω = p[1]
    A = p[2]
    for i ∈ eachindex(du)
        du[i] = 2π * ω[i] * cos(2π * ω[i] * t)
    end
    _du = A * du
    for i ∈ eachindex(du)
        du[i] = _du[i]
    end
end

## IVP
n_dims = 4 # dimensions of observation
tspan = (0.0, 2π)

ω = rand(n_dims)
# ω = 1.0:n_dims

# A = -0.30 * ones(n_dims, n_dims)
A = rand(n_dims, n_dims)
for i ∈ 1:n_dims
    A[i, i] = 1.0
end
# A = I(n_dims)

p = (ω, A)
u0 = 2.0 .* rand(n_dims) .- 1.0

prob = ODEProblem(wave_model!, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6);

##
plot(sol)

