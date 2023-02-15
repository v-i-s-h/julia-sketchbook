# Source: https://julialang.org/blog/2021/10/DEQ/

##
using Flux
using DiffEqSensitivity
using SteadyStateDiffEq
using OrdinaryDiffEq
using Plots
using LinearAlgebra

##
struct DeepEquilibriumNetwork{M, P, RE, A, K}
    model::M
    p::P
    re::RE
    args::A
    kwargs::K
end

Flux.@functor DeepEquilibriumNetwork

function DeepEquilibriumNetwork(model, args...; kwargs...)
    p, re = Flux.destructure(model)
    return DeepEquilibriumNetwork(model, p, re, args, kwargs)
end

Flux.trainable(deq::DeepEquilibriumNetwork) = (deq.p, )

function (deq::DeepEquilibriumNetwork)(x::AbstractArray{T}, p=deq.p) where {T}
    z = deq.re(p)(x) # restructure the model and do forward
    dudt(u, _p, t) = deq.re(_p)(u .+ x) .- u
    ssprob = SteadyStateProblem(ODEProblem(dudt, z, (zero(T), one(T)), p))
    w = solve(ssprob, deq.args...; u0 = z, deq.kwargs...).u
    return w
end

##
device = cpu

ann = Chain(Dense(1, 5), Dense(5, 1)) |> device
deq = DeepEquilibriumNetwork(ann, DynamicSS(Tsit5(), abstol=1f-2, reltol=1f-2))

## Run the DEQ model for regression
X = reshape(collect(1f0:10f0), 1, :) |> device
Y = 2X

opt = ADAM(0.05)

loss(x, y) = sum(abs2, y - deq(x))

## Train
epochs = 1000
for i ∈ 1:epochs
    Flux.train!(loss, Flux.params(deq), ((X, Y), ), opt)
    println(deq([-5] |> device), loss(X, Y)) # debug print for model prediction
end

## Visualizing
function construct_iterator(deq::DeepEquilibriumNetwork, x, p=deq.p)
    executions = 1
    model = deq.re(p) # rebuild model
    previous_value = nothing
    function iterator()
        z = model((executions == 1 ? zero(x) : previous_value) .+ x)
        executions += 1
        previous_value = z
        return z
    end
    return iterator
end

function generate_model_trajectory(deq, x, max_depth::Int, abstol::T=1e-8, 
                                    reltol::T=1e-8) where {T}
    deq_fun = construct_iterator(deq, x)
    values = [x, deq_fun()]
    for i ∈ 2:max_depth
        sol = deq_fun()
        push!(values, sol)
        if (norm(sol - values[end-1]) ≤ abstol) || 
                (norm(sol - values[end-1]) / norm(values[end-1]) ≤ reltol)
            return values
        end
    end
    return values
end

traj = generate_model_trajectory(deq, rand(1, 10) .* 10 |> device, 100)

plot(0:(length(traj) - 1), vcat(traj...) |> cpu, xlabel="Depth", ylabel="Value",
     legend=false)

##