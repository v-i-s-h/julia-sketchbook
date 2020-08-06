#=
# Variational Autoencoder
# Source: https://github.com/FluxML/model-zoo/blob/master/vision/vae_mnist/vae_mnist.jl
=#

using Base.Iterators: partition
using BSON
using CUDAapi: has_cuda_gpu
using DrWatson: struct2dict
using Flux
using Flux: chunk
using Flux.Losses: logitbinarycrossentropy
using Flux.Data: DataLoader
using Images
using Logging: with_logger
using MLDatasets
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using TensorBoardLogger: TBLogger, tb_overwrite
using Random


# Load MNIST data and return loader
function get_data(batch_size)
    xtrain, _ = MLDatasets.MNIST.traindata(Float32)
    # @show size(xtrain) size(ytrain)
    xtrain = reshape(xtrain, 28^2, :)
    DataLoader(xtrain, batchsize=batch_size, shuffle=true)
end


# Define encoder as a Struct followed by a functor
struct Encoder
    linear
    μ
    logσ
    Encoder(input_dim, latent_dim, hidden_dim, device) = new(
        Dense(input_dim, hidden_dim, tanh) |> device,   # linear
        Dense(hidden_dim, latent_dim) |> device,        # μ
        Dense(hidden_dim, latent_dim) |> device         # logσ
    )
end

function (encoder::Encoder)(x)
    h = encoder.linear(x)
    encoder.μ(h), encoder.logσ(h)
end


# Define decoder as a function
Decoder(input_dim, latent_dim, hidden_dim, device) = Chain(
    Dense(latent_dim, hidden_dim, tanh),
    Dense(hidden_dim, input_dim)
) |> device


# function to reconstruct input
function reconstruct(encoder, decoder, x, device)
    μ, logσ = encoder(x)
    z = μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
    μ, logσ, decoder(z)
end

# function for model loss
function model_loss(encoder, decoder, λ, x, device)
    μ, logσ, x̂ = reconstruct(encoder, decoder, x, device)
    len = size(x)[end]

    # compute KL-divergence
    kl_p_q = 0.50f0 * sum(@.(exp(2f0 * logσ) + μ^2 - 1f0 - 2f0 * logσ)) / len
    
    logp_x_z = -logitbinarycrossentropy(x̂, x)

    # regularization
    reg = λ * sum(x -> sum(x.^2), Flux.params(decoder))

    # negative ELBO
    - logp_x_z + kl_p_q + reg
end


# convert the one dimensional output to 2D image
function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(sigmoid.(x |> cpu), y_size), 28, :)...), (2, 1)))
end


# arguments for the train function
@with_kw mutable struct Args
    η = 1e-3
    λ = 0.01f0
    batch_size = 128
    sample_size = 10
    epochs = 20
    seed = 0
    cuda = true
    input_dim = 28^2
    latent_dim = 2
    hidden_dim = 500
    verbose_freq = 10
    tblogger = false
    save_path = "zoo"
end


# train function
function train(; kws...)
    # load hyper params
    args = Args(; kws...)
    args.seed > 0 && Random.seed(args.seed)

    # GPU config
    if args.cuda && has_cuda_gpu()
        device = gpu
        @info "Training in GPU"
    else
        device = cpu
        @info "Training in CPU"
    end

    # Load MNIST images
    loader = get_data(args.batch_size)

    # Initialize encoder and decoder
    encoder = Encoder(args.input_dim, args.latent_dim, args.hidden_dim, device)
    decoder = Decoder(args.input_dim, args.latent_dim, args.hidden_dim, device)

    opt = ADAM(args.η)

    # parameters
    ps = Flux.params(encoder.linear, encoder.μ, encoder.logσ, decoder)

    !ispath(args.save_path) && mkpath(args.save_path)
    
    # logging using TensorBoard.jl
    if args.tblogger
        tblogger = TBLogger(args.save_path, tb_overwrite)
    end

    # fixed input
    original = first(get_data(args.sample_size^2))
    original = original |> device
    image = convert_to_image(original, args.sample_size)
    image_path = joinpath(args.save_path, "original.png")
    save(image_path, image)

    # training
    train_steps = 0
    @info "Start training, total $(args.epochs) epochs"
    for epoch in 1:args.epochs
        @info "Epoch $epoch / $(args.epochs)"
        progress = Progress(length(loader))

        for x in loader
            loss, back = Flux.pullback(ps) do
                model_loss(encoder, decoder, args.λ, x |> device, device)
            end
            ∇ = back(1f0)
            Flux.Optimise.update!(opt, ps, ∇)
            next!(progress; showvalues=[(:loss, loss)])
        end
        
        train_steps += 1

        # save image
        _, _, original_recon = reconstruct(encoder, decoder, original, device)
        image = convert_to_image(original_recon, args.sample_size)
        image_path = joinpath(args.save_path, "epoch_$(epoch).png")
        save(image_path, image)
        @info "Image saved: $image_path"
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
