# Multi Layer Perceptron using Flux.jl

using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDAapi
using MLDatasets

if has_cuda()
    @info "CUDA is available"
    import CuArrays
    CuArrays.allowscalar(false)
end

@with_kw mutable struct Args
    η::Float64 = 3e-4
    batchsize::Int = 1024
    epochs::Int = 5
    device::Function = gpu  # Set as gpu is gpu is available
end

function getdata(args)
    # Load
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)
    # print(size(xtrain), size(xtest))  # => (28, 28, 60000), (28, 28, 10000)

    # Flatten the images
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    # One hot encode the labels
    ytrain = onehotbatch(ytrain, 0:9)
    ytest = onehotbatch(ytest, 0:9)

    # Batching
    train_data = DataLoader(xtrain, ytrain, batchsize=args.batchsize, shuffle=true)
    test_data = DataLoader(xtest, ytest, batchsize=args.batchsize)

    return train_data, test_data
end

function build_model(; imgsize=(28, 28, 1), nclasses=10)
    return Chain(
        Dense(prod(imgsize), 32, relu),
        Dense(32, nclasses)
    )
end

function loss_all(dataloader, model)
    l = 0f0
    for (x, y) in dataloader
        l += logitcrossentropy(model(x), y)
    end
    return l / length(dataloader)
end

function accuracy(dataloader, model)
    acc = 0.0
    for (x, y) in dataloader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y))) * 1 / size(x, 2)
    end
    return acc / length(dataloader)
end

function train(; kws...)
    # Initialize model parameters
    args = Args(;kws...)

    # Load data
    train_data, test_data = getdata(args)

    # Build model
    m = build_model()
    train_data = args.device.(train_data)
    test_data = args.device.(test_data)
    m = args.device(m)
    loss(x, y) = logitcrossentropy(m(x), y)

    # Training
    evalcb = () -> @show(loss_all(train_data, m))
    opt = ADAM(args.η)
    for i in 1:args.epochs
        Flux.train!(loss, params(m), train_data, opt)
        l = loss_all(train_data, m)
        println("Epoch = ", i, "    l = ", l)
    end
    # @epochs args.epochs Flux.train!(loss, params(m), train_data, opt, cb=evalcb)

    # Evaluation
    @show accuracy(train_data, m)
    @show accuracy(test_data, m)
end

cd(@__DIR__)
train()