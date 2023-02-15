# SPSA + NN for data fitting

using Lux, NNlib, Functors, Zygote, Optimisers, ComponentArrays, Statistics
using Random, Plots
using Printf

##
epochs = 2500
log_interval = 50

opt = Optimisers.Adam(0.01f0)

## Create data
function create_dataset(; N=100)
    M = round(Int, N / 4)
    
    # Generate artificial data
    x1s = rand(Float32, M) * 4.5f0
    x2s = rand(Float32, M) * 4.5f0
    xt1s = Array([[x1s[i] + 0.5f0; x2s[i] + 0.5f0] for i in 1:M])
    x1s = rand(Float32, M) * 4.5f0
    x2s = rand(Float32, M) * 4.5f0
    append!(xt1s, Array([[x1s[i] - 5.0f0; x2s[i] - 5.0f0] for i in 1:M]))

    x1s = rand(Float32, M) * 4.5f0
    x2s = rand(Float32, M) * 4.5f0
    xt0s = Array([[x1s[i] + 0.5f0; x2s[i] - 5.0f0] for i in 1:M])
    x1s = rand(Float32, M) * 4.5f0
    x2s = rand(Float32, M) * 4.5f0
    append!(xt0s, Array([[x1s[i] - 5.0f0; x2s[i] + 0.5f0] for i in 1:M]))

    # Store all the data for later
    xs = hcat(xt1s..., xt0s...)
    ts = collect([ones(Float32, 2 * M); zeros(Float32, 2 * M)]')

    return (xs, ts)
end

##
xs, ys = create_dataset(; N=100)

scatter(xs[1, :], xs[2, :]; marker_z=ys, clim=(0, 1))

## Create model
nn = Chain(Dense(2, 3, tanh), Dense(3, 2, tanh), Dense(2, 1, sigmoid))
ps, st = Lux.setup(Random.default_rng(), nn)
ps = ComponentArray(ps)
np = Lux.parameterlength(nn)

println("NN model has $(np) parameters")

##
xx_grid = hcat(
    reshape(
        [[x1, x2] for x1 ∈ -10.0f0:0.1f0:+10.0f0, x2 ∈ -10.0f0:0.1f0:+10.0f0],
        :, 1
    )...
)

## Draw random predictions from model
ypred = nn(xs, ps, st)[1] |> vec
scatter(xs[1, :], xs[2, :]; marker_z=ypred, clim=(0, 1), label=false)

## Draw decision region
ypred_grid = nn(xx_grid, ps, st)[1] |> vec
scatter(xx_grid[1, :], xx_grid[2, :]; marker_z=ypred_grid, 
        marker=:square, markersize=3, markerstrokewidth=0, markeralpha=0.10,
        clim=(0, 1), label=false)

## Loss function to train
function bce_loss(y, ypred)
    return - mean(
        @. y * log(ypred + eps(eltype(ypred))) +
            (1 - y) * log(1 - (ypred + eps(eltype(ypred))))
    )
end

accuracy(y, ypred) = mean(@. y == (ypred > 0.50))

## Train model -- full batch
opt_st = Optimisers.setup(opt, ps)

loss_fn(ps, x, y) = bce_loss(y, nn(x, ps, st)[1])

## Train model using gradients
train_log = Dict(
    "epoch" => [],
    "loss" => [],
    "acc" => []
)

ypred, st = nn(xs, ps, st)
@info "Untrained" loss=loss_fn(ps, xs, ys) accuracy=accuracy(ys, ypred)

for i ∈ 1:epochs
    global opt_st, st, ps, ypred
    gs = gradient(loss_fn, ps, xs, ys)[1] # Compute gradients
    opt_st, ps = Optimisers.update(opt_st, ps, gs)

    if i % log_interval == 0
        ypred, _ = nn(xs, ps, st)
        loss = loss_fn(ps, xs, ys)
        acc = accuracy(ys, ypred) 
        @info "Epoch#$(i)" loss=loss accuracy=acc
        
        append!(train_log["epoch"], i)
        append!(train_log["loss"], loss)
        append!(train_log["acc"], acc)
    end
end

# Save old parameters
ps_trained = deepcopy(ps);

## Plot train
p1 = plot(train_log["epoch"], train_log["loss"], title="Loss", label=false)
p2 = plot(train_log["epoch"], train_log["acc"], title="Acc", label=false)
plot(p1, p2, layout=(2, 1), size=(640, 480))

## Plot decision region and predictions
ypred_grid = nn(xx_grid, ps, st)[1] |> vec
scatter(xx_grid[1, :], xx_grid[2, :]; marker_z=ypred_grid, 
        marker=:square, markersize=3, markerstrokewidth=0, markeralpha=0.10,
        clim=(0, 1), label=false)

ypred = nn(xs, ps, st)[1] |> vec
scatter!(xs[1, :], xs[2, :], marker_z=ypred, 
    marker=:diamond, clim=(0, 1), label=false)

## SPSA
mutable struct SPSA{T}
    t::Int
    a::T
    A::T
    c::T
    α::T
    γ::T
end

function SPSA(a::T, A::T, c::T, α::T, γ::T) where T <: AbstractFloat
    return SPSA(0, a, A, c, α, γ)
end

function (spsa::SPSA{T})(fn, fn_args::Tuple, 
    θ::AbstractArray{T}) where T <: AbstractFloat
    spsa.t += 1 # Update time index

    n = length(θ)
    
    Δ = 2 .* (rand(Float32, n) .> 0.50f0) .- 1

    aₖ = spsa.a / (spsa.t + spsa.A)^spsa.α
    cₖ = spsa.c / spsa.t^spsa.γ

    Δθ = cₖ * Δ
    θ⁺ = θ + Δθ
    θ⁻ = θ - Δθ

    # Compute losses
    l⁺ = fn(θ⁺, fn_args...)
    l⁻ = fn(θ⁻, fn_args...)
    # @info "SPSA" l⁺ l⁻ Δθ

    # println(θ, θ⁺, Δ)

    ĝ = (l⁺ - l⁻) ./ (2 * Δθ)

    # return aₖ * ĝ
    return ĝ
end

function Base.show(io::IO, ::MIME"text/plain", spsa::SPSA)
    @printf(io, 
        "SPSA(t = %d, a = %.3f, A = %.3f, α = %.3f, c = %.3f, γ = %.3f)",
        spsa.t, spsa.a, spsa.A,
        spsa.α, spsa.c, spsa.γ
    )
end

##
spsa = SPSA(10.0f0, 1.0f0, 0.10f0, 0.90f0, 0.18f0)

ps, st = Lux.setup(Random.default_rng(), nn)
ps = ComponentArray(ps)

opt_st = Optimisers.setup(opt, ps)

train_log_spsa = Dict(
    "epoch" => [],
    "loss" => [],
    "acc" => []
)


for i ∈ 1:epochs
    global ps, ypred, opt_st
    ∇ = spsa(loss_fn, (xs, ys), ps) 
    opt_st, ps = Optimisers.update(opt_st, ps, ∇)
    if i % log_interval == 0
        ypred, _ = nn(xs, ps, st)
        loss = loss_fn(ps, xs, ys)
        acc = accuracy(ys, ypred) 
        @info "Epoch#$(i)" loss=loss accuracy=acc
        
        append!(train_log_spsa["epoch"], i)
        append!(train_log_spsa["loss"], loss)
        append!(train_log_spsa["acc"], acc)

    end
end

## Plot train
p1 = plot(train_log["epoch"], train_log["loss"], title="Loss", label="SGD")
p2 = plot(train_log["epoch"], train_log["acc"], title="Acc", label="SGD")
plot!(p1, train_log_spsa["epoch"], train_log_spsa["loss"], title="Loss", label="SPSA")
plot!(p2, train_log_spsa["epoch"], train_log_spsa["acc"], title="Acc", label="SPSA")
plot(p1, p2, layout=(2, 1), size=(640, 480))

