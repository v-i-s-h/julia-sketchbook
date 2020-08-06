#=
Variational AutoEncoder in Flux
Source: https://alecokas.github.io/julia/flux/vae/2020/07/22/convolutional-vae-in-flux.html
=#

using Flux
using Flux: logitbinarycrossentropy
using Flux.Data: DataLoader
using Zygote
using MLDatasets: FashionMNIST
using ImageFiltering: padarray, Fill
using CSV
using BSON: @save
using DataFrames: DataFrame
using ProgressMeter: Progress, next!
using Random


function get_train_loader(batch_size, shuffle::Bool)
    train_x, train_y = FashionMNIST.traindata(Float32)
    train_x = reshape(train_x, (28, 28, 1, :))
    train_x = parent(padarray(train_x, Fill(0, (2, 2, 0, 0))))
    return DataLoader(train_x, train_y, batchsize=batch_size, shuffle=shuffle,
                        partial=false)
end

function save_model(encoder_μ, encoder_logvar, decoder, savedir::String, epoch::Int)
    print("Saving model...")
    let encoder_μ = cpu(encoder_μ), encoder_logvar = cpu(encoder_logvar), decoder = cpu(decoder)
        @save joinpath(savedir, "model-$epoch.bson") encoder_μ encoder_logvar decoder
    end
    println("Done")
end

struct Reshape
    shape
end
Reshape(args...) = Reshape(args)
(r::Reshape)(x) = reshape(x, r.shape)
Flux.@functor Reshape ()

function create_vae()
    # define encoder and decoder
    encoder_features = Chain(
        Conv((4, 4), 1 => 32, relu; stride = 2, pad = 1),
        Conv((4, 4), 32 => 32, relu; stride = 2, pad = 1),
        Conv((4, 4), 32 => 32, relu; stride = 2, pad =1),
        Flux.flatten,
        Dense(32 * 4 * 4, 256, relu),
        Dense(256, 256, relu)
    )

    encoder_μ = Chain(encoder_features, Dense(256, 10))
    encoder_logvar = Chain(encoder_features, Dense(256, 10))

    decoder = Chain(
        Dense(10, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 32 * 4 * 4, relu),
        Reshape(4, 4, 32, :),
        ConvTranspose((4, 4), 32 => 32, relu; stride = 2, pad = 1),
        ConvTranspose((4, 4), 32 => 32, relu; stride = 2, pad = 1),
        ConvTranspose((4, 4), 32 => 1; stride = 2, pad = 1)
    )
    return encoder_μ, encoder_logvar, decoder
end

function vae_loss(encoder_μ, encoder_logvar, decoder, x, β, λ)
    batch_size = size(x)[end]
    @assert batch_size != 0

    μ = encoder_μ(x)
    logvar = encoder_logvar(x)
    z = μ + randn(Float32, size(logvar)) .* exp.(0.50f0 * logvar)
    x_bar = decoder(z)

    # reconstruction loss
    logp_x_z = -sum(logitbinarycrossentropy.(x_bar, x)) / batch_size
    # kl loss
    kl_q_p = 0.50f0 * sum(@.(exp(logvar) + μ^2 - logvar - 1f0)) / batch_size
    # weight decay regularization
    reg = λ * sum(x -> sum(x.^2), Flux.params(encoder_μ, encoder_logvar, decoder))
    # elbo -- to maximize
    elbo = logp_x_z - β .* kl_q_p
    # objective to minimize
    return -elbo + reg
end

function train(encoder_μ, encoder_logvar, decoder, dataloader, num_epochs,
                λ, β, optimizer, savedir)
    trainable_params = Flux.params(encoder_μ, encoder_logvar, decoder)

    for epoch_num ∈ 1:num_epochs
        acc_loss = 0.0
        pbar = Progress(length(dataloader), 1, "Training epoch $epoch_num: ")
        for (x_batch, y_batch) in data_loader
            loss, back = pullback(trainable_params) do
                vae_loss(encoder_μ, encoder_logvar, decoder, x_batch, β, λ)
            end
            gradients = back(1f0)
            Flux.Optimise.update!(optimizer, trainable_params, gradients)
            if isnan(loss)
                break
            end
            acc_loss += loss
            next!(pbar; showvalues=[(:loss, loss)])
        end
        @assert length(dataloader) > 0
        avg_loss = acc_loss / length(dataloader)
        metrics = DataFrame(epoch=epoch_num, negative_elbo=avg_loss)
        println(metrics)
    end
    println("Training complete.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running...")
    # define hyper parameters
    batch_size = 64
    shuffle_data = true
    η = 0.00001
    β = 1f0
    λ = 0.01f0
    num_epochs = 10
    save_dir = "results"
    data_loader = get_train_loader(batch_size, shuffle_data)
    encoder_μ, encoder_logvar, decoder = create_vae()
    train(encoder_μ, encoder_logvar, decoder, data_loader, num_epochs, 
            λ, β, ADAM(η), save_dir)
else
    println("Seriously.?")
    @show @__FILE__
    @show PROGRAM_FILE
end
