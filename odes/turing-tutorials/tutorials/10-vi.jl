# Variational Inference

using Random
using Turing
using Turing: Variational

Random.seed!(42);

## generate data
x = randn(2000);

##
@model function vi_sample_model(x)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0.0, sqrt(s))
    for i in axes(x, 1)
        x[i] ~ Normal(m, √(s))
    end
end

m = vi_sample_model(x)

##
sample_nuts = sample(m, NUTS(200, 0.65), 10000)

##
advi = ADVI(10, 1000)
q = vi(m, advi)

samples = rand(q, 10000);
##
using Plots, LaTeXStrings, StatsPlots

p1 = histogram(samples[1, :]; bins=100, normed=true, alpha=0.2, color=:blue, label="")
density!(samples[1, :]; label="s (ADVI)", color=:blue, linewidth=2)
density!(sample_nuts, :s; label="s (NUTS)", color=:green, linewidth=2)
vline!([var(x)]; label="s (data)", color=:black)
vline!([mean(samples[1, :])]; color=:blue, label="")

p2 = histogram(samples[2, :]; bins=100, normed=true, alpha=0.2, color=:blue, label="")
density!(samples[2, :]; label="m (ADVI)", color=:blue, linewidth=2)
density!(sample_nuts, :m; label="m (NUTS)", color=:green, linewidth=2)
vline!([mean(x)]; label="m (data)", color=:black)
vline!([mean(samples[2, :])]; color=:blue, label="")

plot(p1, p2; layout=(2, 1), size=(900, 500))

## # Linear regression
Random.seed!(1);
using RDatasets
using MLDataUtils: shuffleobs, splitobs, rescale!
using LinearAlgebra
using Distances

Turing.setprogress!(false);

## data
data = RDatasets.dataset("datasets", "mtcars")

select!(data, Not(:Model))
trainset, testset = splitobs(data, at=0.70)
μₓ, σₓ = rescale!(trainset)
rescale!(testset, propertynames(trainset), μₓ, σₓ)

target_name = :MPG
train = Matrix(select(trainset, Not(target_name)))
train_label = trainset[:, target_name]
test = Matrix(select(testset, Not(target_name)))
test_label = testset[:, target_name]

##
@model function linear_regression(x, y, n_obs, n_vars, ::Type{T}=Vector{Float64}) where {T}
    σ² ~ truncated(Normal(0, 100); lower=0)

    α ~ Normal(0, 3)
    β ~ MvNormal(zeros(n_vars), 10 * I)

    μ  = α .+ x * β
    return y ~ MvNormal(μ, σ²)
end

n_obs, n_vars = size(train)
m = linear_regression(train, train_label, n_obs, n_vars)

## Perform VI
q₀ = Variational.meanfield(m)
advi = ADVI(10, 10000)
opt = Variational.DecayedADAGrad(1e-2, 1.1, 0.9)
q = vi(m, advi, q₀; optimizer=opt)

##
z = rand(q, 10000);
avg = vec(mean(z; dims=2))

_, sym2range = bijector(m, Val(true))

##
function  plot_variational_marginals(z, sym2range)
    ps = []

    for (i, sym) ∈ enumerate(keys(sym2range))
        indices = union(sym2range[sym]...)
        if sum(length.(indices)) > 1
            offset = 1
            for r ∈ indices
                for j in r
                    p = density(
                        z[j, :];
                        title="$(sym)[$offset]", titlefont=10, label=""
                    )
                    push!(ps, p)

                    offset += 1
                end
            end
        else
            p = density(z[first(indices), :]; title="$(sym)", titlefont=10, label="")
            push!(ps, p)
        end
    end

    return plot(ps...; layout=(length(ps), 1), size=(500, 1500))
end

plot_variational_marginals(z, sym2range)


# Compare to NUTS
chain = sample(m, NUTS(0.65), 10000)

plot(chain)

##
function prediction_chain(chain, x)
    p = get_params(chain)
    α = mean(p.α)
    β = collect(mean.(p.β))
    return α .+ x * β
end

function prediction_vi(samples, sym2ranges, x)
    α = mean(samples[union(sym2ranges[:α]...)])
    β = vec(mean(samples[union(sym2ranges[:β]...), :]; dims=2))
    return α .+ x * β
end

function rescale_back(x, μₓ, σₓ)
    return x * σₓ .+ μₓ
end

##
pred_vi_train = prediction_vi(z, sym2range, train) |> x -> rescale_back(x, μₓ[1], σₓ[1])
pred_vi_test = prediction_vi(z, sym2range, test) |> x -> rescale_back(x, μₓ[1], σₓ[1])

pred_nuts_train = prediction_chain(chain, train) |> x -> rescale_back(x, μₓ[1], σₓ[1])
pred_nuts_test = prediction_chain(chain, test) |> x -> rescale_back(x, μₓ[1], σₓ[1])

train_label_rescaled = rescale_back(train_label, μₓ[1], σₓ[1])
test_label_rescaled = rescale_back(test_label, μₓ[1], σₓ[1])

##
println(
"Training set",
    "\n\tVI loss : ", msd(pred_vi_train, train_label_rescaled),
    "\n\tBayes loss : ", msd(pred_nuts_train, train_label_rescaled),
"\nTest set",
    "\n\tVI loss : ", msd(pred_vi_test, test_label_rescaled),
    "\n\tBayes loss : ", msd(pred_nuts_test, test_label_rescaled)
)

##
z = rand(q, 1000)
preds = hcat(
    [prediction_vi(z[:, i], sym2range, test) |> x -> rescale_back(x, μₓ[1], σₓ[1])
        for i ∈ 1:size(z, 2)]...
);

scatter(
    1:size(test, 1),
    mean(preds; dims=2);
    yerr =std(preds; dims=2),
    label="Prediction (μ ± σ)",
    size=(900, 500),
    merkersize=8
)
scatter!(1:size(test, 1), test_label_rescaled, label="True")
xaxis!(1:size(test, 1))
ylims!(0, 50)
title!("Mean field ADVI (Normal)")

##
preds = hcat(
    [prediction_chain(chain[i], test) |> x -> rescale_back(x, μₓ[1], σₓ[1])
        for i ∈ 1:10:size(chain, 1)]...
);

scatter(
    1:size(test, 1),
    mean(preds; dims=2);
    yerr =std(preds; dims=2),
    label="Prediction (μ ± σ)",
    size=(900, 500),
    merkersize=8
)
scatter!(1:size(test, 1), test_label_rescaled, label="True")
xaxis!(1:size(test, 1))
ylims!(0, 50)
title!("MCMC (NUTS)")

## Alternative
using Bijectors
using Bijectors: Scale, Shift

##
d = length(q)
base_dist = Turing.DistributionsAD.TuringDiagMvNormal(zeros(d), ones(d))

##
to_constrained = inv(bijector(m))

##
function getq(θ)
    d = length(θ) ÷ 2
    A = @inbounds θ[1:d]
    b = @inbounds θ[(d + 1):(2 * d)]

    b = to_constrained ∘ Shift(b; dim=Val(1)) ∘ Scale(exp.(A); dim=Val(1))

    return transformed(base_dist, b)
end

##
q_mf_normal = vi(m, advi, getq, randn(2 * d))

##
p1 = plot_variational_marginals(rand(q_mf_normal, 10000), sym2range)
p2 = plot_variational_marginals(rand(q, 10000), sym2range)

plot(p1, p2; layout=(1, 2), size=(800, 2000))

## Relaxing mean-field assumption
using LinearAlgebra
using ComponentArrays, UnPack

proto_arr = ComponentArray(; L=zeros(d, d), b=zeros(d))
proto_axes = getaxes(proto_arr)
num_params = length(proto_arr)

function getq(θ)
    L, b = begin
        @unpack L, b = ComponentArray(θ, proto_axes)
        LowerTriangular(L), b
    end
    D = Diagonal(diag(L))
    A = L - D + exp(D)

    b = to_constrained ∘ Shift(b; dim=Val(1)) ∘ Scale(A; dim=Val(1))

    return transformed(base_dist, b)
end

##
advi = ADVI(10, 20000)
q_full_normal = vi(
    m, advi, getq, randn(num_params);
    optimizer=Variational.DecayedADAGrad(1e-2)
)

##
A = q_full_normal.transform.ts[1].a

##
