# Source: https://julialang.org/blog/2021/10/DEQ/

## DEQ to classify the digits of MNIST

using Zygote
using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using ProgressMeter: @showprogress
import MLDatasets
using DiffEqSensitivity
using SteadyStateDiffEq
using OrdinaryDiffEq
using LinearAlgebra
using Plots
using MultivariateStats
using Statistics
using ColorSchemes

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

Flux.trainable(deq::DeepEquilibriumNetwork) = (deq.p,)

function (deq::DeepEquilibriumNetwork)(x::AbstractArray{T}, p=deq.p) where {T}
    z = deq.re(p)(x)
    dudt(u, _p, t) = deq.re(_p)(u .+ x) .- u
    ssprob = SteadyStateProblem(ODEProblem(dudt, z, (zero(T), one(T)), p))
    return solve(ssprob, deq.args...; u0=z, deq.kwargs...).u
end
##

function Net()
    return Chain(
        Flux.flatten,
        Dense(784, 100),
        DeepEquilibriumNetwork(
            Chain(Dense(100, 500, tanh), Dense(500, 100)),
            DynamicSS(Tsit5(), abstol=1f-1, reltol=1f-1)
        ),
        Dense(100, 10)
    )
end

##
function get_data(args)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    device = cpu
    xtrain = reshape(xtrain, 28, 28, 1, :) |> device
    xtest = reshape(xtest, 28, 28, 1, :) |> device
    ytrain = onehotbatch(ytrain, 0:9) |> device
    ytest = onehotbatch(ytest, 0:9) |> device

    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, 
                              shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=args.batchsize)

    return train_loader, test_loader
end


function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x = x |> device
        y = y |> device
        ŷ = model(x)

        l += Flux.Losses.logitcrossentropy(ŷ, y) * size(x)[end]
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end

    return (loss = l / ntot |> round4, acc = acc / ntot * 100 |> round4)
end

round4(x) = round(x, digits=4)

##

Base.@kwdef mutable struct Args
    η = 3e-4    # learning rate
    λ = 0       # L2 reg
    batchsize = 8  #
    epochs = 1
    seed = 0
end

##
function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    device = cpu

    # Data
    train_loader, test_loader = get_data(args)
    # @info "Dataset MNIST $(train_loader.nobs) train and $(test_loader.nobs) test samples"
    
    # model
    model = Net() |> device
    ps = Flux.params(model)

    opt = ADAM(args.η)
    if args.λ > 0
        opt = Optimiser(opt, WeightDecay(args.λ))
    end

    # Training
    @info "Start training"
    for epoch ∈ 1:args.epochs
        @showprogress for (x, y) ∈ train_loader
            x = x |> device
            y = y |> device

            gs = Flux.gradient(
                () -> Flux.Losses.logitcrossentropy(model(x), y), ps
            )
            Flux.Optimise.update!(opt, ps, gs)
        end
        loss, accuracy = eval_loss_accuracy(test_loader, model, device)
        println("Epoch: $epoch || Test loss: $loss || Acc: $accuracy")
    end

    return model, train_loader, test_loader
end

##
model, train_loader, test_loader = train(batchsize=128, epochs=1);

##

